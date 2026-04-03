"""
Codex Provider Adapter (OpenAI ChatGPT Backend)

Adapter for using OpenAI models via ChatGPT Plus/Pro subscription OAuth,
hitting the same private backend API that the Codex CLI and OpenClaw use:
``https://chatgpt.com/backend-api/codex/responses``

This is the OpenAI equivalent of ClaudeMaxAdapter — subscription-included
usage, not API key billing.

The protocol is the standard OpenAI Responses API but served from the
ChatGPT backend with OAuth Bearer auth + chatgpt-account-id header.

Requirements:
- Active ChatGPT Plus/Pro subscription
- `codex login` completed (stores token in ~/.codex/auth.json)
  OR CODEX_AUTH_TOKEN env var set
"""
import base64
import json
import logging
import platform
from typing import Any, Dict, List, Optional, AsyncIterator, Type, Union

import httpx
from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .model_metadata import ModelInfo, ModelCategory

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _extract_account_id(token: str) -> str:
    """Extract chatgpt_account_id from the JWT access token claims."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token")
        # Decode JWT payload (base64url)
        payload_b64 = parts[1]
        # Add padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No chatgpt_account_id in token claims")
        return account_id
    except Exception as e:
        raise ValueError(f"Failed to extract account ID from token: {e}") from e


def _build_headers(token: str, account_id: str) -> dict:
    """Build request headers matching the OpenClaw/Codex protocol."""
    ua = f"kestrel ({platform.system()} {platform.release()}; {platform.machine()})"
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "User-Agent": ua,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }


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


def _build_request_body(
    model: str,
    input_messages: list,
    instructions: Optional[str] = None,
    tools: Optional[list] = None,
    stream: bool = True,
    **kwargs,
) -> dict:
    """Build Responses API request body matching the ChatGPT backend protocol."""
    body: Dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": stream,
        "input": input_messages,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
    if "max_tokens" in kwargs:
        body["max_output_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        body["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        body["top_p"] = kwargs["top_p"]
    return body


async def _parse_sse_events(response: httpx.Response):
    """Parse SSE events from an httpx streaming response."""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


class CodexAdapter(LLMAdapter):
    """
    Adapter for OpenAI Codex subscription via ChatGPT backend.

    Uses httpx to hit chatgpt.com/backend-api/codex/responses with
    OAuth Bearer token + chatgpt-account-id header. The response format
    is the standard OpenAI Responses API SSE stream.
    """

    def __init__(self):
        self.name = "codex"

    async def get_response(
        self,
        client: Any,  # OAuth token string stored as client
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """Get a response from the ChatGPT backend Responses API.

        The ChatGPT backend requires stream=true, so we consume the
        SSE stream internally and assemble the final response.
        """
        token = client  # Provider registry stores token as "client"
        if not isinstance(token, str):
            raise RuntimeError(
                "Codex adapter requires an OAuth token. "
                "Run `codex login` or set CODEX_AUTH_TOKEN."
            )

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        responses_tools = _convert_tools_to_responses_format(tools)

        body = _build_request_body(
            model=model,
            input_messages=input_messages,
            instructions=instructions,
            tools=responses_tools,
            stream=True,  # ChatGPT backend requires streaming
            **kwargs,
        )

        # Consume SSE stream and assemble final response
        content_parts: List[str] = []
        parsed_tool_calls = None
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    logger.error(f"Codex API error {resp.status_code}: {error_text}")
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")

                    if event_type == "response.output_text.delta":
                        content_parts.append(event.get("delta", ""))

                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            idx = event.get("output_index", 0)
                            func_calls[idx] = {
                                "id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "response.function_call_arguments.delta":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] += event.get("delta", "")

                    elif event_type == "response.function_call_arguments.done":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] = event.get("arguments", "")

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        final_usage = resp_data.get("usage", {})

        content = "".join(content_parts) if content_parts else None

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

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw=None,
            input_tokens=final_usage.get("input_tokens"),
            output_tokens=final_usage.get("output_tokens"),
            total_tokens=final_usage.get("total_tokens"),
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
        """Get a streaming response from the ChatGPT backend."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)

        body = _build_request_body(
            model=model,
            input_messages=input_messages,
            instructions=instructions,
            stream=True,
            **kwargs,
        )

        logger.info(f"Starting Codex stream for model: {model}")
        chunk_count = 0

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    logger.error(f"Codex stream error {resp.status_code}: {error_text}")
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")
                    if event_type == "response.output_text.delta":
                        chunk_count += 1
                        yield event.get("delta", "")

        logger.info(f"Codex stream completed. Total chunks: {chunk_count}")

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Stream response with tool call detection."""
        token = client
        if not isinstance(token, str):
            raise RuntimeError("Codex adapter requires an OAuth token.")

        account_id = _extract_account_id(token)
        headers = _build_headers(token, account_id)
        instructions, input_messages = _extract_instructions_and_input(messages)
        responses_tools = _convert_tools_to_responses_format(tools)

        body = _build_request_body(
            model=model,
            input_messages=input_messages,
            instructions=instructions,
            tools=responses_tools,
            stream=True,
            **kwargs,
        )

        logger.info(f"Starting Codex stream with tools for model: {model}")
        text_content = ""
        chunk_count = 0
        func_calls: Dict[int, Dict[str, str]] = {}
        final_usage: Dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream("POST", CODEX_BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    error_text = resp.text[:500]
                    raise RuntimeError(
                        f"Codex API returned {resp.status_code}: {error_text}"
                    )

                async for event in _parse_sse_events(resp):
                    event_type = event.get("type", "")

                    if event_type == "response.output_text.delta":
                        chunk_count += 1
                        delta = event.get("delta", "")
                        text_content += delta
                        yield delta

                    elif event_type == "response.function_call_arguments.delta":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] += event.get("delta", "")

                    elif event_type == "response.function_call_arguments.done":
                        idx = event.get("output_index", 0)
                        if idx not in func_calls:
                            func_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        func_calls[idx]["arguments"] = event.get("arguments", "")

                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            idx = event.get("output_index", 0)
                            func_calls[idx] = {
                                "id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif event_type == "response.completed":
                        resp_data = event.get("response", {})
                        usage = resp_data.get("usage", {})
                        final_usage = usage

        # Yield final tool call response if any
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
                input_tokens=final_usage.get("input_tokens"),
                output_tokens=final_usage.get("output_tokens"),
                total_tokens=final_usage.get("total_tokens"),
            )

        logger.info(
            f"Codex stream completed. Text chunks: {chunk_count}, "
            f"Tool calls: {len(func_calls)}"
        )

    async def list_models(self) -> List[ModelInfo]:
        """Return available models for Codex subscription."""
        return [
            ModelInfo(
                id="gpt-5.4",
                display_name="GPT-5.4 (Codex)",
                provider="codex",
                category=ModelCategory.CHAT,
                context_limit=1_050_000,
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
