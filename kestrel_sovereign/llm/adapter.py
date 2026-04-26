"""
Base LLM Adapter

Provides a standardized interface for interacting with different LLM providers.
Supports:
- Streaming and non-streaming responses
- Structured output (Pydantic models)
- Vision capabilities (images)
- Tool/function calling (OpenAI format)
- API-based model discovery
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, AsyncIterator, TYPE_CHECKING
import logging
import os
from pydantic import BaseModel
from dataclasses import dataclass

from .image_utils import process_images

if TYPE_CHECKING:
    from .model_metadata import ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """
    Unified response from an LLM adapter.

    Attributes:
        content: Text content of the response (may be None if tool_calls present)
        tool_calls: List of tool calls requested by the model
        raw: The raw response object from the provider (for debugging)
        input_tokens: Number of tokens in the prompt/input (uncached portion)
        output_tokens: Number of tokens in the completion/output
        total_tokens: Total tokens used (input + output, excluding cache reads)
        cache_creation_input_tokens: Tokens written to the prompt cache this call
        cache_read_input_tokens: Tokens read from the prompt cache this call
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw: Any = None
    # Usage tracking for billing/metering
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Anthropic prompt-cache breakdown. Either may be 0 (no cache write/read on
    # this call); both will be None for providers that don't report cache usage.
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

    @property
    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


class LLMAdapter(ABC):
    """
    Base class for LLM adapters.

    Standardizes the interface for sending prompts and receiving responses
    from different LLM providers (OpenAI, Ollama, Anthropic, etc.).
    """

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Creates a message list in OpenAI format.

        Args:
            user_prompt: The user's query
            system_prompt: Optional system prompt
            images: Optional list of images (file paths, base64, or bytes)

        Returns:
            List of message dictionaries
        """
        messages = []
        user_prompt_content = []

        if user_prompt:
            user_prompt_content.append({"type": "text", "text": user_prompt})

        self._handle_images(images, user_prompt_content)

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt_content:
            messages.append({"role": "user", "content": user_prompt_content})

        return messages

    @abstractmethod
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
        Get a response from the LLM.

        Args:
            client: The provider-specific client
            model: Model name to use
            messages: Chat messages
            format: Response format (e.g., "json") - DEPRECATED, use response_format
            tools: Optional list of tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output.
                When provided, the LLM response will be validated against this schema.
            **kwargs: Additional provider-specific parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with content and/or tool calls
        """
        pass

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
        Get a streaming response from the LLM.

        Default implementation raises NotImplementedError.
        Override in subclasses that support streaming.

        Args:
            client: The provider-specific client
            model: Model name to use
            messages: Chat messages
            tools: Optional tools (note: streaming with tools is provider-specific)
            response_format: Optional Pydantic model for structured output.
                Note: Streaming with structured output may not be supported by all providers.
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks as they arrive
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    async def list_models(self) -> List["ModelInfo"]:
        """
        List available models from this provider.

        Default implementation raises NotImplementedError.
        Override in subclasses to call the provider's models API.

        Returns:
            List of ModelInfo objects with model metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support model listing")

    def contribute_system_prompt(
        self, model_id: str, base: Optional[str]
    ) -> Optional[str]:
        """Augment a system prompt with provider/model-specific contributions.

        Default returns ``base`` unchanged. Subclasses override to inject
        behavior contracts, format hints, or other model-family discipline
        that does not belong in the universal system prompt. The contribution
        must be byte-stable across turns for any given ``model_id`` so that
        the prefix-cache invariant from #703 / #706 is preserved.

        See ``gpt5_overlay.prepend_gpt5_overlay`` for the canonical example.
        """
        return base

    def _apply_system_prompt_contribution(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
    ) -> List[Dict[str, Any]]:
        """Apply ``contribute_system_prompt`` to a chat-completions message list.

        Returns a new list — does not mutate the input. The first ``system``-role
        message has its content replaced by ``contribute_system_prompt(model_id,
        original)``. If no system message is present and the contribution is
        non-empty, a new system message is prepended.
        """
        new_messages: List[Dict[str, Any]] = []
        augmented = False
        for msg in messages:
            if not augmented and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    contributed = self.contribute_system_prompt(model_id, content)
                    if contributed != content:
                        new_messages.append({**msg, "content": contributed})
                        augmented = True
                        continue
            new_messages.append(msg)

        if not augmented:
            contributed = self.contribute_system_prompt(model_id, None)
            if contributed:
                return [{"role": "system", "content": contributed}, *new_messages]
        return new_messages

    def _handle_images(
        self,
        images: Optional[List[Union[str, bytes]]],
        user_prompt_content: List[Dict[str, Any]]
    ) -> None:
        """
        Handle images using centralized image_utils with auto-resize.

        Uses OpenAI's 2048x2048 limit for the base adapter (OpenAI-format messages).
        Provider-specific adapters override or pass their own provider limits.
        """
        if not images:
            return

        # Use centralized image processing with auto-resize for OpenAI limits
        for processed in process_images(images, provider="openai"):
            user_prompt_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{processed.mime_type};base64,{processed.data}"
                }
            })
