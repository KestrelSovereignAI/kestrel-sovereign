"""
Codex Provider Adapter (OpenAI OAuth)

Adapter for using OpenAI models via Codex OAuth subscription authentication,
similar to how ClaudeMaxAdapter wraps Anthropic with OAuth token auth.

Uses the OpenAI Responses API (not Chat Completions) since that's what
Codex models are optimized for. The OAuth access_token from `codex login`
is passed as api_key to the OpenAI SDK — both are sent as Bearer tokens.

Requirements:
- pip install openai
- Active ChatGPT Plus/Pro subscription
- `codex login` completed (stores token in ~/.codex/auth.json)
  OR CODEX_AUTH_TOKEN env var set

How it works:
1. OAuth token from codex login is passed as api_key to the OpenAI SDK
2. Both API keys and OAuth tokens are sent as `Authorization: Bearer <token>`
3. Responses API is used instead of Chat Completions for codex-optimized models
"""
import json
import logging
from typing import Any, Dict, List, Optional, AsyncIterator, Type, Union

import openai
from pydantic import BaseModel

from .adapter import LLMResponse, ToolCall
from .model_metadata import ModelInfo, ModelCategory
from .openai_adapter import OpenAIAdapter
from .retry import with_retry

logger = logging.getLogger(__name__)


def _extract_instructions_and_input(messages):
    """Split messages into instructions (system prompt) and input messages."""
    instructions = None
    input_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            instructions = content
        else:
            input_messages.append(msg)
    return instructions, input_messages


def _convert_tools_to_responses_format(tools):
    """Convert OpenAI function calling format to Responses API tool format."""
    if not tools:
        return None
    responses_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            responses_tools.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
    return responses_tools or None


class CodexAdapter(OpenAIAdapter):
    """
    Adapter for OpenAI Codex subscription using OAuth token auth.

    Subclasses OpenAIAdapter — uses the Responses API instead of
    Chat Completions. Authentication is handled at client creation
    time in the provider registry (OAuth token passed as api_key).
    """

    def __init__(self):
        self.name = "codex"

    async def get_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """Get a response using the OpenAI Responses API."""
        try:
            if client is None:
                raise RuntimeError("Codex adapter requires an OpenAI client but none is configured")

            instructions, input_messages = _extract_instructions_and_input(messages)

            extra_kwargs: Dict[str, Any] = {}
            if instructions:
                extra_kwargs["instructions"] = instructions

            responses_tools = _convert_tools_to_responses_format(tools)
            if responses_tools:
                extra_kwargs["tools"] = responses_tools

            if "max_tokens" in kwargs:
                extra_kwargs["max_output_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                extra_kwargs["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                extra_kwargs["top_p"] = kwargs["top_p"]

            response = await with_retry(
                client.responses.create,
                model=model,
                input=input_messages,
                **extra_kwargs
            )

            # Extract text and tool calls from output items
            content = None
            parsed_tool_calls = None

            for item in response.output:
                if item.type == "message":
                    texts = []
                    for part in item.content:
                        if part.type == "output_text":
                            texts.append(part.text)
                    if texts:
                        content = "\n".join(texts)

                elif item.type == "function_call":
                    if parsed_tool_calls is None:
                        parsed_tool_calls = []
                    try:
                        args = json.loads(item.arguments) if item.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw": item.arguments}
                    parsed_tool_calls.append(ToolCall(
                        id=item.id,
                        name=item.name,
                        arguments=args,
                    ))

            # Extract usage
            input_tokens = None
            output_tokens = None
            total_tokens = None
            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = response.usage.total_tokens

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except openai.AuthenticationError as e:
            logger.error(
                f"Codex authentication failed: {e}. "
                "Run `codex login` to refresh your OAuth token, "
                "or check CODEX_AUTH_TOKEN env var."
            )
            raise
        except openai.RateLimitError as e:
            logger.error(f"Codex rate limit exceeded: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"Codex connection error: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"Codex API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Codex adapter failed: {e}", exc_info=True)
            raise

    async def get_streaming_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Get a streaming response using the Responses API."""
        try:
            instructions, input_messages = _extract_instructions_and_input(messages)

            extra_kwargs: Dict[str, Any] = {}
            if instructions:
                extra_kwargs["instructions"] = instructions
            if "max_tokens" in kwargs:
                extra_kwargs["max_output_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                extra_kwargs["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                extra_kwargs["top_p"] = kwargs["top_p"]

            logger.info(f"Starting Codex Responses API stream for model: {model}")
            stream = await with_retry(
                client.responses.create,
                model=model,
                input=input_messages,
                stream=True,
                **extra_kwargs
            )

            chunk_count = 0
            async for event in stream:
                if event.type == "response.output_text.delta":
                    chunk_count += 1
                    yield event.delta

            logger.info(f"Codex stream completed. Total chunks: {chunk_count}")

        except openai.AuthenticationError as e:
            logger.error(
                f"Codex authentication failed during streaming: {e}. "
                "Run `codex login` to refresh your OAuth token."
            )
            raise
        except openai.RateLimitError as e:
            logger.error(f"Codex rate limit exceeded during streaming: {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"Codex connection error during streaming: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"Codex API error during streaming: {e}")
            raise
        except Exception as e:
            logger.error(f"Codex streaming failed: {e}", exc_info=True)
            raise

    async def get_streaming_response_with_tools(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Stream response with tool call detection via Responses API."""
        try:
            instructions, input_messages = _extract_instructions_and_input(messages)

            extra_kwargs: Dict[str, Any] = {}
            if instructions:
                extra_kwargs["instructions"] = instructions

            responses_tools = _convert_tools_to_responses_format(tools)
            if responses_tools:
                extra_kwargs["tools"] = responses_tools

            if "max_tokens" in kwargs:
                extra_kwargs["max_output_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                extra_kwargs["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                extra_kwargs["top_p"] = kwargs["top_p"]

            logger.info(f"Starting Codex stream with tools for model: {model}")
            stream = await with_retry(
                client.responses.create,
                model=model,
                input=input_messages,
                stream=True,
                **extra_kwargs
            )

            text_content = ""
            chunk_count = 0
            func_calls: Dict[int, Dict[str, str]] = {}

            async for event in stream:
                if event.type == "response.output_text.delta":
                    chunk_count += 1
                    text_content += event.delta
                    yield event.delta

                elif event.type == "response.function_call_arguments.delta":
                    idx = event.output_index
                    if idx not in func_calls:
                        func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    func_calls[idx]["arguments"] += event.delta

                elif event.type == "response.function_call_arguments.done":
                    idx = event.output_index
                    if idx not in func_calls:
                        func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    func_calls[idx]["arguments"] = event.arguments

                elif event.type == "response.output_item.added":
                    item = event.item
                    if hasattr(item, "type") and item.type == "function_call":
                        idx = event.output_index
                        func_calls[idx] = {
                            "id": getattr(item, "id", "") or "",
                            "name": getattr(item, "name", "") or "",
                            "arguments": "",
                        }

                elif event.type == "response.completed":
                    resp = event.response
                    input_tokens = None
                    output_tokens = None
                    total_tokens = None
                    if hasattr(resp, "usage") and resp.usage:
                        input_tokens = resp.usage.input_tokens
                        output_tokens = resp.usage.output_tokens
                        total_tokens = resp.usage.total_tokens

                    if func_calls:
                        parsed_tool_calls = []
                        for idx in sorted(func_calls.keys()):
                            fc = func_calls[idx]
                            try:
                                args = json.loads(fc["arguments"]) if fc["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {"raw": fc["arguments"]}
                            parsed_tool_calls.append(ToolCall(
                                id=fc["id"],
                                name=fc["name"],
                                arguments=args,
                            ))

                        yield LLMResponse(
                            content=text_content if text_content else None,
                            tool_calls=parsed_tool_calls,
                            raw=None,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                        )

            logger.info(
                f"Codex stream completed. Text chunks: {chunk_count}, "
                f"Tool calls: {len(func_calls)}"
            )

        except openai.AuthenticationError as e:
            logger.error(f"Codex auth failed during streaming with tools: {e}")
            raise
        except openai.RateLimitError as e:
            logger.error(f"Codex rate limit during streaming with tools: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"Codex API error during streaming with tools: {e}")
            raise
        except Exception as e:
            logger.error(f"Codex streaming with tools failed: {e}", exc_info=True)
            raise

    async def list_models(self) -> List[ModelInfo]:
        """
        Return available models for Codex subscription.

        Hardcoded because OAuth tokens lack the api.model.read scope
        needed for model discovery via API.
        """
        return [
            ModelInfo(
                id="gpt-5.4",
                display_name="GPT-5.4 (Codex)",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                is_featured=True,
            ),
            ModelInfo(
                id="gpt-5.4-mini",
                display_name="GPT-5.4 Mini (Codex)",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                is_featured=True,
            ),
            ModelInfo(
                id="gpt-5.3-codex",
                display_name="GPT-5.3 Codex",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-5.2-codex",
                display_name="GPT-5.2 Codex",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-5.1-codex-max",
                display_name="GPT-5.1 Codex Max",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                id="gpt-5.1-codex-mini",
                display_name="GPT-5.1 Codex Mini",
                provider="codex",
                category=ModelCategory.CHAT,
                supports_tools=True,
                supports_streaming=True,
            ),
        ]
