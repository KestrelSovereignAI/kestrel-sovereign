"""
Google Vertex AI Adapter

Adapter for Google Cloud Vertex AI using the google-genai SDK with support for:
- Tool/function calling
- Vision (image inputs)
- Streaming responses
- API-based model discovery
- Exponential backoff for rate limiting

Uses the new google-genai SDK (not deprecated google-generativeai or google-cloud-aiplatform).
Authentication via Application Default Credentials (ADC).
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type

from pydantic import BaseModel

from .adapter import LLMAdapter, LLMResponse, ToolCall
from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)
from .model_metadata import ModelInfo, ModelCategory
from .retry import with_retry
from .image_utils import process_images
from .genai_custody import close_genai_client, genai_operation, genai_stream, joined_iterator, prepare_genai_client
from .google_adapter import _finish_reason, _normalized_google_genai_usage
from .gemini_contents import (
    SYSTEM_MARKER_ROLE,
    to_gemini_contents,
    tool_call_from_part,
    tool_result_messages,
)
from .output_ceiling import attach_stop_reason

logger = logging.getLogger(__name__)





@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI adapter."""
    project_id: str
    location: str = "us-central1"
    credentials_file: Optional[str] = None


class VertexAIAdapter(LLMAdapter):
    """
    Adapter for Google Vertex AI using google-genai SDK.

    Key differences from GoogleAdapter (the Gemini API with an API key):
    - Uses google-genai SDK with vertexai=True flag
    - Requires GCP project authentication (ADC or service account)
    - Supports both Gemini models and partner models (Claude on Vertex, Llama on Vertex)
    - Provides API-based model discovery via client.aio.models.list()
    """

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_embeddings=True,
            structured_output_mode=StructuredOutputMode.PROVIDER_NATIVE,
            tool_streaming_mode=ToolStreamingMode.NONSTREAM_FALLBACK,
            vision_input_mode=VisionInputMode.GEMINI_INLINE_DATA,
            embedding_model="text-embedding-004",
            embedding_dim=768,
            model_dependent=("tools", "vision", "structured_output"),
            notes=(
                "Structured output uses Vertex/Gemini response_schema.",
                "Streaming tool calls use the framework's non-streaming fallback path.",
            ),
        )

    @staticmethod
    def _embedding_values(item: Any) -> Optional[List[float]]:
        if isinstance(item, dict):
            values = item.get("values")
        else:
            values = getattr(item, "values", None)
        return list(values) if values is not None else None

    @classmethod
    def _embeddings_from_response(cls, response: Any, count: int) -> List[Optional[List[float]]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        out: List[Optional[List[float]]] = [None] * count
        for idx, item in enumerate(embeddings or []):
            if idx >= count:
                break
            out[idx] = cls._embedding_values(item)
        return out

    @genai_operation
    async def aembed(
        self,
        client: Any,
        text: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        genai_client = client if client else self._get_client()
        response = await genai_client.aio.models.embed_content(
            model=model or "text-embedding-004",
            contents=text,
        )
        return self._embeddings_from_response(response, 1)[0]

    @genai_operation
    async def aembed_batch(
        self,
        client: Any,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []
        genai_client = client if client else self._get_client()
        response = await genai_client.aio.models.embed_content(
            model=model or "text-embedding-004",
            contents=texts,
        )
        return self._embeddings_from_response(response, len(texts))

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        credentials_file: Optional[str] = None
    ):
        """
        Initialize Vertex AI adapter.

        Args:
            project_id: GCP project ID (or use GOOGLE_CLOUD_PROJECT/GCP_PROJECT_ID env var)
            location: GCP region (default: us-central1)
            credentials_file: Optional path to service account JSON (or use GOOGLE_APPLICATION_CREDENTIALS)
        """
        # Resolve project ID from multiple sources
        self.project_id = (
            project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
            or self._get_project_from_credentials()
        )
        # Env var takes precedence over parameter for location
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION") or location
        # Resolve credentials file
        self.credentials_file = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self._client = None
        self._closed = False

    async def aclose(self):
        """Retire only this adapter's cached client, never a supplied route."""
        self._closed = True
        if self._client is not None:
            await close_genai_client(self._client)

    def _get_project_from_credentials(self) -> Optional[str]:
        """Extract project_id from GOOGLE_APPLICATION_CREDENTIALS file if available."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            try:
                import json
                with open(creds_path, encoding='utf-8') as f:
                    creds = json.load(f)
                    return creds.get("project_id")
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load Google Cloud credentials from {creds_path}: {e}", exc_info=True)
                # Return None instead of raising - project_id is optional
                return None
            except Exception as e:
                logger.error(f"Unexpected error loading Google Cloud credentials: {e}", exc_info=True)
                return None
        return None

    def _get_client(self):
        """
        Get or create the google-genai client.

        Lazy initialization to avoid import errors if SDK not installed.
        Prefers GOOGLE_API_KEY (AI Studio) over Vertex AI service account.
        """
        if self._closed:
            raise RuntimeError("Vertex adapter is closed")
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )

            # Prefer API key mode (AI Studio) if GOOGLE_API_KEY is set
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                logger.info("Using Google AI Studio (API key mode)")
                self._client = genai.Client(api_key=api_key)
                return self._client

            # Fall back to Vertex AI mode (service account)
            if not self.project_id:
                raise ValueError(
                    "Vertex AI requires a project ID. "
                    "Set GOOGLE_CLOUD_PROJECT env var or pass project_id."
                )

            logger.info(f"Using Vertex AI (project: {self.project_id}, location: {self.location})")
            client_kwargs = {
                "vertexai": True,
                "project": self.project_id,
                "location": self.location,
            }

            # Optional: Add explicit credentials if provided
            if self.credentials_file:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client_kwargs["credentials"] = credentials

            self._client = genai.Client(**client_kwargs)

        return self._client

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create messages in Vertex AI / Gemini format.

        Vertex AI uses 'contents' with role and parts, same as Gemini.
        System prompts are passed separately via system_instruction parameter.
        """
        messages = []

        # Note: System prompt should be passed via system_instruction parameter
        # in generate_content, not in messages. We store it for later extraction.
        if system_prompt:
            # Store as metadata - will be extracted in get_response
            messages.append({
                "role": SYSTEM_MARKER_ROLE,  # Special marker, not sent to API
                "parts": [{"text": system_prompt}]
            })

        # Add actual user prompt with optional images
        if user_prompt or images:
            parts = []

            if user_prompt:
                parts.append({"text": user_prompt})

            # Handle images using centralized image_utils with auto-resize
            # Vertex AI has 3072x3072 limit
            if images:
                for processed in process_images(images, provider="vertex_ai"):
                    parts.append({
                        "inline_data": {
                            "mime_type": processed.mime_type,
                            "data": processed.data
                        }
                    })

            messages.append({"role": "user", "parts": parts})

        return messages

    def _convert_tools_to_vertex_format(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Vertex AI format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Vertex AI format (function declarations):
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

    def _generation_config(
        self,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Type[BaseModel]],
        call_kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """The request config both ``generate_content`` paths send.

        Vertex takes tools on the config as function declarations, next to the
        structured-output settings (#2811); with no tools the config carries
        no ``tools`` key. ``None`` when there is nothing to configure.
        """
        config: Dict[str, Any] = {}
        if call_kwargs.get("max_tokens"):
            config["max_output_tokens"] = call_kwargs["max_tokens"]
        if "temperature" in call_kwargs:
            config["temperature"] = call_kwargs["temperature"]
        if system_prompt:
            config["system_instruction"] = system_prompt

        # Handle structured output via response_schema
        if response_format is not None and issubclass(response_format, BaseModel):
            # Gemini supports JSON schema for structured output
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_format.model_json_schema()

        tools_config = None
        if tools:
            function_declarations = self._convert_tools_to_vertex_format(tools)
            if function_declarations:
                tools_config = [{"function_declarations": function_declarations}]
        if tools_config is not None:
            config["tools"] = tools_config

        return config or None

    @staticmethod
    def _tool_call(part: Any, index: int) -> Optional[ToolCall]:
        """The tool call a response part carries, or None if it carries none.

        Parsed by :func:`tool_call_from_part`, which keeps a Gemini 3 call's
        thought signature (#3619). ``index`` numbers the call within its
        response.
        """
        return tool_call_from_part(part, f"vertex_call_{index}")

    @genai_operation
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
        Get response from Vertex AI.

        Args:
            client: The genai.Client instance (or None to use internal client)
            model: Model name (e.g., 'gemini-2.0-flash-001', 'gemini-1.5-pro')
            messages: Chat messages in OpenAI format, or already in Gemini
                format (``parts``); converted by ``to_gemini_contents``
            format: Response format (ignored for Vertex AI)
            tools: Optional tools in OpenAI format (will be converted)
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature)

        Returns:
            LLMResponse with content and/or tool calls
        """
        try:
            # Use provided client or internal client
            genai_client = client if client else self._get_client()

            # OpenAI-format messages become Gemini contents (#3608)
            system_prompt, contents = to_gemini_contents(messages)

            config = self._generation_config(
                system_prompt, tools, response_format, kwargs,
            )

            # Generate content using the async client with retry
            response = await with_retry(
                genai_client.aio.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )

            # Parse response
            content = None
            parsed_tool_calls = None
            finish_reason = None

            # Check for function calls in candidates
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = _finish_reason(candidate)
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content = part.text
                            continue
                        tool_call = self._tool_call(part, len(parsed_tool_calls or ()))
                        if tool_call is not None:
                            parsed_tool_calls = parsed_tool_calls or []
                            parsed_tool_calls.append(tool_call)

            # Fallback: try to get text directly
            if content is None and hasattr(response, 'text'):
                content = response.text

            # Extract token usage from response
            # Gemini usage_metadata, normalized once for both Gemini routes
            input_tokens = output_tokens = total_tokens = None
            cache_read_input_tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_input_tokens,
                ) = _normalized_google_genai_usage(response.usage_metadata)

            return attach_stop_reason(LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason)

        except Exception as e:
            logger.error(f"Vertex AI API error: {e}")
            raise

    @genai_stream
    async def _stream_with_usage(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Stream text chunks then emit one terminal LLMResponse with usage.

        Shared by :meth:`get_streaming_response` (which filters the terminal
        response out to preserve its text-only contract) and the no-tools
        branch of :meth:`get_streaming_response_with_tools` (which forwards it
        so the service layer can meter streamed turns — #1684). Vertex sends
        ``usage_metadata`` cumulatively across the stream; the latest non-empty
        value carries the final counts.
        """
        try:
            # Use provided client or internal client
            genai_client = client if client else self._get_client()

            # OpenAI-format messages become Gemini contents (#3608)
            system_prompt, contents = to_gemini_contents(messages)

            config = self._generation_config(
                system_prompt, tools, response_format, kwargs,
            )

            # Generate streaming content with retry - await the coroutine first to get the async iterator
            stream = await with_retry(
                genai_client.aio.models.generate_content_stream,
                model=model,
                contents=contents,
                config=config,
            )
            text_content = ""
            usage_meta = None
            finish_reason = None
            tool_calls: List[ToolCall] = []
            async for chunk in stream:
                if getattr(chunk, "usage_metadata", None):
                    usage_meta = chunk.usage_metadata
                # The finish reason arrives on the last chunk's candidate.
                candidates = getattr(chunk, "candidates", None)
                if candidates:
                    finish_reason = _finish_reason(candidates[0]) or finish_reason
                    # With tools on the request, a function call can arrive
                    # on the stream; carry it on the terminal response.
                    candidate_content = getattr(candidates[0], "content", None)
                    for part in getattr(candidate_content, "parts", None) or ():
                        tool_call = self._tool_call(part, len(tool_calls))
                        if tool_call is not None:
                            tool_calls.append(tool_call)
                if hasattr(chunk, 'text') and chunk.text:
                    text_content += chunk.text
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        # A candidate cut before any text (its thinking took
                        # the whole budget) has content but no parts (#3621).
                        content = getattr(candidate, "content", None)
                        for part in getattr(content, "parts", None) or ():
                            if getattr(part, "text", None):
                                text_content += part.text
                                yield part.text

            input_tokens = output_tokens = total_tokens = None
            cache_read_input_tokens = None
            if usage_meta is not None:
                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_input_tokens,
                ) = _normalized_google_genai_usage(usage_meta)
            yield attach_stop_reason(LLMResponse(
                content=text_content if text_content else None,
                tool_calls=tool_calls or None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason)

        except Exception as e:
            logger.error(f"Vertex AI streaming error: {e}")
            raise

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
        Get streaming response from Vertex AI (text-only contract).

        Delegates to :meth:`_stream_with_usage` and drops the terminal
        usage-bearing :class:`LLMResponse` so existing callers keep their
        ``AsyncIterator[str]`` contract. A function call on that response
        cannot travel on a text stream; its drop is logged, never silent.

        Yields:
            Text chunks as they arrive
        """
        async with joined_iterator(self._stream_with_usage(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            **kwargs
        )) as stream:
            async for item in stream:
                if isinstance(item, str):
                    yield item
                elif item.tool_calls:
                    logger.warning(
                        "Vertex AI returned %d function call(s) (%s) on the "
                        "text-only stream, which cannot carry them; use "
                        "get_streaming_response_with_tools to receive tool calls",
                        len(item.tool_calls),
                        ", ".join(call.name for call in item.tool_calls),
                    )

    async def get_streaming_response_with_tools(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """
        Stream response with tool call detection.

        FALLBACK STRATEGY: Vertex AI streaming with tool detection is complex
        and the SDK support varies. This method uses a fallback approach:
        1. If tools are provided, make a non-streaming call first to detect tool calls
        2. If tool calls are detected, yield the LLMResponse immediately
        3. Otherwise, stream the text response

        This provides a consistent interface across all providers.

        Args:
            client: The genai.Client instance (or None to use internal client)
            model: Model name
            messages: Chat messages
            tools: Optional tools in OpenAI format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Terminal response at end-of-stream carrying token usage (and tool_calls when present)
        """
        try:
            logger.info(f"Starting Vertex AI stream with tools for model: {model}")

            # If tools are provided, check for tool calls first via non-streaming
            if tools:
                response = await self.get_response(
                    client=client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    **kwargs
                )

                # If tool calls detected, yield the response immediately
                if response.has_tool_calls:
                    logger.info(f"Vertex AI detected {len(response.tool_calls)} tool calls")
                    yield response
                    return

                # No tool calls - yield the text content, then a terminal
                # LLMResponse carrying usage so the service layer meters this
                # turn (#1684). `response` already holds token counts from the
                # non-streaming probe; the terminal response is read only for
                # usage (content was already streamed above).
                if response.content:
                    yield response.content
                yield response
                return

            # No tools - stream text and forward the terminal usage response so
            # the service layer meters text-only streamed turns (#1684).
            async with joined_iterator(self._stream_with_usage(
                client=client,
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs
            )) as stream:
                async for item in stream:
                    yield item

        except Exception as e:
            logger.error(f"Vertex AI streaming with tools failed: {e}")
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

        ``messages`` ends with the assistant turn whose ``tool_calls`` the
        ``tool_results`` (``{"tool_call_id", "content"}``) answer; each result
        becomes a ``function_response`` named after its call (#3608).
        """
        return await self.get_response(
            client=client,
            model=model,
            messages=[*messages, *tool_result_messages(tool_results)],
            tools=tools,
            **kwargs
        )

    @genai_operation(resolve_client=False)
    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """List available models from Vertex AI.

        A supplied route client is borrowed, as for :meth:`get_response`.
        Without one, use the adapter's project/location-bound cached client.

        Uses the google-genai SDK's asynchronous ``aio.models.list()``. The
        listing awaits, so the per-route discovery deadline can cancel it
        (#3646); the synchronous ``models.list()`` it used blocked the event
        loop, so an endpoint that never answered stalled every other route's
        discovery and every discovery deadline with it.

        Returns:
            List of ModelInfo objects for each available model
        """
        try:
            genai_client = client if client is not None else self._get_client()
            prepare_genai_client(genai_client)
            models = []

            async for model in await genai_client.aio.models.list():
                model_id = model.name if hasattr(model, 'name') else str(model)
                display_name = getattr(model, 'display_name', None) or model_id
                description = getattr(model, 'description', None)

                # Detect model category
                category = ModelCategory.CHAT
                lower_id = model_id.lower()
                if "embed" in lower_id:
                    category = ModelCategory.EMBEDDING
                elif "image" in lower_id or "imagen" in lower_id:
                    category = ModelCategory.IMAGE

                models.append(ModelInfo(
                    id=model_id,
                    provider="vertex_ai",
                    display_name=display_name,
                    category=category,
                    description=description,
                    context_limit=getattr(model, 'input_token_limit', None),
                    supports_vision=any(x in lower_id for x in ["gemini", "vision"]),
                    supports_tools=True,
                    supports_streaming=True,
                ))

            logger.info(f"Discovered {len(models)} Vertex AI models")
            return models

        except Exception as e:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            if is_execution_control_error(e):
                raise
            logger.warning(f"Failed to discover Vertex AI models: {e}")
            # Return fallback models
            return self._get_fallback_models()

    def _get_fallback_models(self) -> List[ModelInfo]:
        """
        Return fallback model list when API discovery fails.

        Reads from the discovery cache for vertex_ai models.
        Falls back to an empty list if no cache available.
        """
        try:
            from .model_catalog import get_catalog_service
            catalog = get_catalog_service()
            cached = catalog.load_cache()
            if cached:
                return [m for m in cached if m.provider == "vertex_ai"]
        except Exception as e:
            logger.warning(f"Could not load vertex_ai fallback models from cache: {e}")
        return []

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def cost_per_1m_tokens(self) -> Optional[Dict[str, float]]:
        # Vertex Gemini pricing tracks the public API closely.
        return {"input": 1.25, "output": 5.00}

    def substrate_type(self) -> Optional[str]:
        return "gemini"

    def display_name(self) -> Optional[str]:
        return "Vertex AI"

    def key_env_var(self) -> Optional[str]:
        # Vertex authenticates via Application Default Credentials, not
        # an env-var API key. None signals "no key-env-var pattern" so
        # the keys UI doesn't prompt for one.
        return None


# Factory function for creating configured adapter
def create_vertex_adapter(
    project_id: Optional[str] = None,
    location: str = "us-central1",
    credentials_file: Optional[str] = None
) -> VertexAIAdapter:
    """
    Create a configured Vertex AI adapter.

    Args:
        project_id: GCP project ID (or use GOOGLE_CLOUD_PROJECT env var)
        location: GCP region (default: us-central1)
        credentials_file: Optional path to service account JSON

    Returns:
        Configured VertexAIAdapter instance
    """
    return VertexAIAdapter(
        project_id=project_id,
        location=location,
        credentials_file=credentials_file
    )
