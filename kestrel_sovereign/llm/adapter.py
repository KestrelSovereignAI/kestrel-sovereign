"""Framework-side LLM adapter base.

The abstract contract — :class:`LLMAdapter`, :class:`LLMResponse`,
:class:`ToolCall` — lives in :mod:`kestrel_sdk.llm` (promoted in
SDK 0.5.0) so third-party provider plugins can depend only on
``kestrel-sovereign-sdk`` without pulling in the framework. This
module re-exports those names for the historical
``kestrel_sovereign.llm.adapter`` import path and adds the in-tree
helpers that depend on framework-only utilities.

In-tree adapters subclass :class:`LLMAdapter` (the framework-enriched
version exported here) to inherit OpenAI-format message construction
and image processing for free. Third-party plugins subclass
:class:`kestrel_sdk.llm.LLMAdapter` directly and bring their own
message construction (or import these helpers when running in-tree).

The framework-only addition is :meth:`LLMAdapter.create_messages`
plus :meth:`LLMAdapter._handle_images`, both of which depend on
:mod:`kestrel_sovereign.llm.image_utils` (PIL-aware image resize and
provider-specific dimension limits). That helper is heavier than the
SDK's pydantic-only-by-default discipline allows, so it stays
framework-side.
"""

from typing import Any, Dict, List, Optional, Union

from kestrel_sdk.llm import LLMResponse, ToolCall
from kestrel_sdk.llm import LLMAdapter as _SDKLLMAdapter

from .image_utils import process_images

__all__ = ["LLMAdapter", "LLMResponse", "ToolCall", "build_messages", "messages_for"]


def build_messages(
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a text-only chat-completions message list in OpenAI format.

    The fallback used by :func:`messages_for` for adapters that don't
    expose a ``create_messages`` method (third-party plugins
    subclassing ``kestrel_sdk.llm.LLMAdapter`` directly). Plain
    ``role`` / ``content`` strings — no images, no provider-specific
    parts. Plugin backends that don't speak OpenAI shape should
    override :meth:`LLMAdapter.create_messages` instead of relying on
    this fallback.

    Args:
        user_prompt: User-role text content, or ``None`` to omit the
            user message.
        system_prompt: System-role text content, or ``None`` to omit
            the system message.

    Returns:
        A list of message dicts ready to pass into ``get_response``.
        Empty if both prompts are ``None``.
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def messages_for(
    adapter: Any,
    *,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build messages for ``adapter``, preferring its provider-specific shape.

    Single entry point used by every text-only call site in the
    framework (``LLMService.generate`` / ``audit`` / ``remote-first``
    / ``get_response_with_model``, the streaming pipeline, council
    deliberation). Dispatches:

    1. If ``adapter`` has a ``create_messages`` method (in-tree
       adapters and any plugin that subclassed the framework's
       enriched ``LLMAdapter`` or overrode the method), call it. This
       preserves provider-native shapes — Gemini ``parts``, Vertex
       ``_system`` markers, Anthropic content blocks — that
       :meth:`LLMAdapter.get_response` expects.
    2. Otherwise (SDK-only plugin subclassing
       ``kestrel_sdk.llm.LLMAdapter`` directly with no
       ``create_messages`` override), fall back to
       :func:`build_messages` for plain OpenAI-shape text messages.
       This is the right default for OpenAI-compatible plugin
       backends (Kimi, DeepSeek, etc., which is the primary intended
       use case for SDK-only plugins). Plugins targeting non-
       OpenAI-shape backends should override ``create_messages`` to
       return their native format.

    Image-bearing message construction stays on the adapter via
    ``create_messages(images=...)`` — there is no fallback for that
    case because image parts are inherently provider-specific.
    """
    if hasattr(adapter, "create_messages"):
        return adapter.create_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    return build_messages(user_prompt=user_prompt, system_prompt=system_prompt)


class LLMAdapter(_SDKLLMAdapter):
    """Framework-enriched LLM adapter base.

    Inherits the abstract contract from :class:`kestrel_sdk.llm.LLMAdapter`
    (``get_response`` abstract; ``get_streaming_response``,
    ``list_models``, ``contribute_system_prompt`` optional) and adds
    OpenAI-format message construction with image handling, which
    depends on the framework's image utilities.

    In-tree adapters should subclass this; third-party plugins should
    subclass :class:`kestrel_sdk.llm.LLMAdapter` directly to keep
    their dependency surface minimal.
    """

    def create_messages(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build a chat-completions message list in OpenAI format.

        Args:
            user_prompt: The user's query. Becomes a single text part
                inside the user-role message's content list.
            system_prompt: Optional system prompt. Becomes a
                system-role message prepended to the conversation.
            images: Optional list of images (file paths, base64
                strings, or raw bytes). Each is processed via
                :func:`process_images` with provider="openai" limits
                and appended to the user message's content list as
                an ``image_url`` part.

        Returns:
            A list of message dicts ready to pass into ``get_response``.
        """
        messages: List[Dict[str, Any]] = []
        user_prompt_content: List[Dict[str, Any]] = []

        if user_prompt:
            user_prompt_content.append({"type": "text", "text": user_prompt})

        self._handle_images(images, user_prompt_content)

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if user_prompt_content:
            messages.append({"role": "user", "content": user_prompt_content})

        return messages

    def _handle_images(
        self,
        images: Optional[List[Union[str, bytes]]],
        user_prompt_content: List[Dict[str, Any]],
    ) -> None:
        """Process images and append them to a user-message content list.

        Uses OpenAI's 2048x2048 limit for the base adapter
        (OpenAI-format messages). Provider-specific adapters can
        override or pass their own provider limits to
        :func:`process_images` if they need different behavior.
        """
        if not images:
            return

        for processed in process_images(images, provider="openai"):
            user_prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed.mime_type};base64,{processed.data}"
                    },
                }
            )
