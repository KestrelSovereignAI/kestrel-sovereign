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

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Union

from kestrel_sdk.llm import LLMResponse, ToolCall
from kestrel_sdk.llm import LLMAdapter as _SDKLLMAdapter

from .image_utils import process_images

@dataclass(frozen=True)
class ThinkingDelta:
    """Provider-separated model reasoning emitted during a stream.

    ``content`` is intentionally not assistant answer text. Agent
    streaming turns it into chat-only UI metadata so storage and
    follow-up context keep the visible response clean.
    """

    content: str
    provider: Optional[str] = None


def split_thinking_from_content(
    content: Optional[str],
    reasoning_content: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(thinking, final_content)`` from provider output.

    Recognises two structural signals: a provider-native
    ``reasoning_content`` field and ``<think>…</think>`` tags inside
    ``content``. Plain prose outside a tag is never reclassified as
    reasoning — if a model leaks reasoning past ``</think>`` it stays
    visible. Rely on llama.cpp's ``--reasoning-format deepseek`` or the
    chat template to wrap reasoning correctly.
    """
    thinking_parts = []
    if isinstance(reasoning_content, str) and reasoning_content:
        thinking_parts.append(reasoning_content)
    if not content:
        return ("\n\n".join(thinking_parts).strip() or None), content

    def collect_think(match: re.Match) -> str:
        thinking_parts.append(match.group(1).strip())
        return ""

    clean = re.sub(
        r'<think>(.*?)</think>\s*',
        collect_think,
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return ("\n\n".join(p for p in thinking_parts if p).strip() or None), clean


class ThinkingContentSplitter:
    """Streaming parser for ``<think>…</think>`` tags in content streams.

    Provider-native ``reasoning_content`` fields are surfaced upstream
    of this splitter; this class only handles inline tag stripping.
    Anything outside a ``<think>`` block passes through as visible
    content unchanged.
    """

    def __init__(self, *, provider: str):
        self.provider = provider
        self.in_think = False
        self.tag_buffer = ""

    def feed(self, text: str):
        if not text:
            return []
        events = []
        self.tag_buffer += text

        while self.tag_buffer:
            lower = self.tag_buffer.lower()
            if self.in_think:
                end = lower.find("</think>")
                if end < 0:
                    keep = 0
                    closing = "</think>"
                    max_check = min(len(self.tag_buffer), len(closing) - 1)
                    for i in range(max_check, 0, -1):
                        if closing.startswith(lower[-i:]):
                            keep = i
                            break
                    thinking = self.tag_buffer[:-keep] if keep else self.tag_buffer
                    if thinking:
                        events.append(ThinkingDelta(thinking, provider=self.provider))
                    self.tag_buffer = self.tag_buffer[-keep:] if keep else ""
                    return events
                thinking = self.tag_buffer[:end]
                if thinking:
                    events.append(ThinkingDelta(thinking, provider=self.provider))
                self.tag_buffer = self.tag_buffer[end + len("</think>"):]
                self.in_think = False
                continue

            start = lower.find("<think>")
            if start < 0:
                keep = 0
                max_check = min(len(self.tag_buffer), len("<think>") - 1)
                for i in range(max_check, 0, -1):
                    if "<think>".startswith(lower[-i:]):
                        keep = i
                        break
                plain = self.tag_buffer[:-keep] if keep else self.tag_buffer
                self.tag_buffer = self.tag_buffer[-keep:] if keep else ""
                if plain:
                    events.append(plain)
                return events

            plain = self.tag_buffer[:start]
            if plain:
                events.append(plain)
            self.tag_buffer = self.tag_buffer[start + len("<think>"):]
            self.in_think = True

        return events

    def flush(self):
        events = []
        if self.tag_buffer:
            if self.in_think:
                events.append(ThinkingDelta(self.tag_buffer, provider=self.provider))
            else:
                events.append(self.tag_buffer)
            self.tag_buffer = ""
        return events


__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "ThinkingDelta",
    "ThinkingContentSplitter",
    "ToolCall",
    "build_messages",
    "messages_for",
    "split_thinking_from_content",
]


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

    #: Whether this adapter reports token usage *incrementally* during a
    #: stream (not only in the terminal ``LLMResponse``). When True, the
    #: service streaming path passes a ``usage_sink`` dict that the adapter
    #: populates as usage events arrive, so a mid-stream abort/timeout can
    #: still flush partial usage to the meter (#1684). Default False: most
    #: providers only surface usage on the final chunk, where the terminal
    #: ``LLMResponse`` already carries it.
    supports_partial_usage_flush: bool = False

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
