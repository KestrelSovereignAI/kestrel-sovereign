"""
Google Gemini Adapter

Adapter for Google's Gemini API with support for:
- Tool/function calling
- Vision (image inputs)
- Streaming responses
- API-based model discovery
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, AsyncIterator

from .adapter import LLMAdapter, LLMResponse, ToolCall
from .model_metadata import ModelInfo, ModelCategory
from .image_utils import process_images

logger = logging.getLogger(__name__)


class GoogleAdapter(LLMAdapter):
    """
    Adapter for Google Gemini API.

    Note: Gemini uses a different message format than OpenAI.
    Uses 'contents' with 'role' and 'parts'.
    """

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create messages in Google Gemini format.

        Gemini uses 'contents' with role and parts.
        System prompts are included as initial user/model exchange.
        """
        messages = []

        # Add system instruction as user message with model acknowledgment
        if system_prompt:
            messages.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            messages.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })

        # Add actual user prompt with optional images
        if user_prompt or images:
            parts = []

            if user_prompt:
                parts.append({"text": user_prompt})

            # Handle images using centralized image_utils
            if images:
                for processed in process_images(images):
                    parts.append({
                        "inline_data": {
                            "mime_type": processed.mime_type,
                            "data": processed.data
                        }
                    })

            messages.append({"role": "user", "parts": parts})

        return messages

    def _convert_tools_to_gemini_format(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Gemini format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Gemini format (function declarations):
        {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
        """
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}})
                })
        return function_declarations

    async def get_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get response from Google Gemini API.

        Args:
            client: Google GenerativeModel instance
            model: Model name (e.g., 'gemini-2.0-flash-exp')
            messages: List of message dicts in Gemini format
            format: Response format (ignored for Gemini)
            tools: Optional tools in OpenAI format (will be converted)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and/or tool calls
        """
        try:
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 8192),
            }

            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]

            # Prepare tool config
            tools_config = None
            if tools:
                tools_config = [{
                    "function_declarations": self._convert_tools_to_gemini_format(tools)
                }]

            # Generate content
            response = await client.generate_content_async(
                contents=messages,
                generation_config=generation_config,
                tools=tools_config
            )

            # Parse response
            content = None
            parsed_tool_calls = None

            # Check for function calls
            candidate = response.candidates[0] if response.candidates else None
            if candidate:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content = part.text
                    elif hasattr(part, 'function_call'):
                        if parsed_tool_calls is None:
                            parsed_tool_calls = []
                        fc = part.function_call
                        # Gemini function_call has name and args
                        args = dict(fc.args) if hasattr(fc, 'args') else {}
                        parsed_tool_calls.append(ToolCall(
                            id=f"gemini_call_{len(parsed_tool_calls)}",
                            name=fc.name,
                            arguments=args
                        ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response
            )

        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise

    async def get_streaming_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get streaming response from Google Gemini.

        Yields:
            Text chunks as they arrive
        """
        try:
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 8192),
            }

            tools_config = None
            if tools:
                tools_config = [{
                    "function_declarations": self._convert_tools_to_gemini_format(tools)
                }]

            response = await client.generate_content_async(
                contents=messages,
                generation_config=generation_config,
                tools=tools_config,
                stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google Gemini streaming error: {e}")
            raise

    async def continue_with_tool_results(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Continue conversation after executing tool calls.

        Gemini expects function responses in a specific format.
        """
        extended_messages = messages.copy()

        # Add function response parts
        function_responses = []
        for result in tool_results:
            function_responses.append({
                "function_response": {
                    "name": result.get("name", "unknown"),
                    "response": {"result": result["content"]}
                }
            })

        extended_messages.append({
            "role": "function",
            "parts": function_responses
        })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools,
            **kwargs
        )

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models from Google Gemini API.

        Uses the google-generativeai SDK's genai.list_models().

        Returns:
            List of ModelInfo objects for each available model
        """
        try:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not set, returning empty model list")
                return []

            try:
                import google.generativeai as genai
            except ImportError:
                logger.warning("google-generativeai not installed, returning empty model list")
                return []

            genai.configure(api_key=api_key)
            models = []

            for model in genai.list_models():
                model_name = model.name if hasattr(model, 'name') else str(model)
                # Remove "models/" prefix if present
                model_id = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
                display_name = getattr(model, 'display_name', model_id)
                description = getattr(model, 'description', None)

                # Detect model category
                category = ModelCategory.CHAT
                lower_id = model_id.lower()
                if "embed" in lower_id:
                    category = ModelCategory.EMBEDDING
                elif "image" in lower_id or "imagen" in lower_id:
                    category = ModelCategory.IMAGE

                # Detect vision support
                supports_vision = "gemini" in lower_id or "vision" in lower_id

                models.append(ModelInfo(
                    id=model_id,
                    provider="google",
                    display_name=display_name,
                    category=category,
                    description=description,
                    supports_vision=supports_vision,
                    supports_tools="gemini" in lower_id,  # Only Gemini models support tools
                    supports_streaming=True,
                ))

            logger.info(f"Google returned {len(models)} models")
            return models

        except Exception as e:
            logger.error(f"Failed to list Google models: {e}")
            return []
