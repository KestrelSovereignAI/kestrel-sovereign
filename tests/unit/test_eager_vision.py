"""Eager vision (#1662 PR B) — pasted/dropped images folded into the model turn.

Covers the three layers the dispatch crosses:
  * base-adapter ``attach_images_to_last_user_message`` / ``_extract_user_text``
    (provider-native injection, delegated to each adapter's ``create_messages``);
  * the LLM service ``_apply_eager_vision`` capability gate (inject for a
    vision model, warn-and-pass-through for a blind one — no silent drop);
  * the agent ``_resolve_eager_images`` (only *inline* images resolve to bytes).
"""
import base64
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock

# Smallest valid image: a 1x1 transparent PNG (so process_images/PIL accept it).
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# --- base adapter text extraction -------------------------------------------

def test_extract_user_text_handles_all_shapes():
    from kestrel_sovereign.llm.adapter import LLMAdapter
    f = LLMAdapter._extract_user_text
    assert f({"role": "user", "content": "hi"}) == "hi"
    assert f({"role": "user", "content": ""}) is None
    assert f({"role": "user", "content": [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {}},
        {"type": "text", "text": "b"},
    ]}) == "a\nb"
    # Gemini "parts" shape
    assert f({"role": "user", "parts": [{"text": "x"}, {"inline_data": {}}]}) == "x"
    assert f({"role": "user", "content": []}) is None


# --- provider-native injection shapes ---------------------------------------

def test_openai_adapter_injects_image_url_block():
    from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
    a = OpenAIAdapter()
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "what is this?"}]
    out = a.attach_images_to_last_user_message(msgs, [_PNG])
    # system untouched; last user becomes a multimodal content list.
    assert out[0] == {"role": "system", "content": "sys"}
    content = out[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # Input not mutated (a fresh list is returned).
    assert msgs[1]["content"] == "what is this?"


def test_anthropic_adapter_injects_content_block():
    from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
    a = AnthropicAdapter()
    out = a.attach_images_to_last_user_message(
        [{"role": "user", "content": "look"}], [_PNG])
    content = out[0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"]


def test_ollama_adapter_injects_images_key():
    from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
    a = OllamaAdapter()
    out = a.attach_images_to_last_user_message(
        [{"role": "user", "content": "see"}], [_PNG])
    # Ollama keeps content a string; images ride a separate key.
    assert out[0]["content"] == "see"
    assert isinstance(out[0].get("images"), list) and out[0]["images"]
    # No stale OpenAI-style content list left behind.
    assert not isinstance(out[0]["content"], list)


def test_attach_targets_last_user_turn_only():
    from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
    a = OpenAIAdapter()
    msgs = [{"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}]
    out = a.attach_images_to_last_user_message(msgs, [_PNG])
    assert out[0]["content"] == "first"            # earlier user untouched
    assert isinstance(out[2]["content"], list)      # only the last user folded
    assert any(p.get("type") == "image_url" for p in out[2]["content"])


def test_attach_noop_without_images():
    from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
    a = OpenAIAdapter()
    msgs = [{"role": "user", "content": "hi"}]
    assert a.attach_images_to_last_user_message(msgs, None) is msgs
    assert a.attach_images_to_last_user_message(msgs, []) is msgs


# --- service-layer capability gate ------------------------------------------

class _VisionAdapter:
    def provider_capabilities(self):
        return MagicMock(supports_vision=True)

    def attach_images_to_last_user_message(self, messages, images):
        return messages + [{"injected": len(images)}]


class _BlindAdapter:
    def provider_capabilities(self):
        return MagicMock(supports_vision=False)

    def attach_images_to_last_user_message(self, messages, images):
        raise AssertionError("must not inject into a non-vision adapter")


def test_apply_eager_vision_injects_for_vision_adapter():
    from kestrel_sovereign.llm.streaming import StreamingMixin
    svc = StreamingMixin.__new__(StreamingMixin)
    msgs = [{"role": "user", "content": "x"}]
    out = svc._apply_eager_vision(_VisionAdapter(), msgs, [b"img"], "openai", "gpt-4o")
    assert out[-1] == {"injected": 1}


def test_apply_eager_vision_warns_and_passes_through_for_blind_adapter(caplog):
    from kestrel_sovereign.llm.streaming import StreamingMixin
    svc = StreamingMixin.__new__(StreamingMixin)
    msgs = [{"role": "user", "content": "x"}]
    with caplog.at_level(logging.WARNING):
        out = svc._apply_eager_vision(_BlindAdapter(), msgs, [b"img"], "codex", "gpt-5")
    # The image is NOT silently mangled into a request the model rejects;
    # the messages pass through untouched and the log says why.
    assert out is msgs
    assert "not vision-capable" in caplog.text


def test_apply_eager_vision_noop_without_images():
    from kestrel_sovereign.llm.streaming import StreamingMixin
    svc = StreamingMixin.__new__(StreamingMixin)
    msgs = [{"role": "user", "content": "x"}]
    assert svc._apply_eager_vision(_BlindAdapter(), msgs, None, "codex", "gpt-5") is msgs


# --- agent eager-image resolution -------------------------------------------

@pytest.mark.asyncio
async def test_resolve_eager_images_resolves_inline_images_only():
    from kestrel_sovereign.agent.streaming import StreamingMixin
    agent = MagicMock()
    agent.storage.retrieve_file = AsyncMock(return_value=b"IMGBYTES")
    resolve = StreamingMixin._resolve_eager_images.__get__(agent)
    h1, h2, h3 = "a" * 64, "b" * 64, "c" * 64
    out = await resolve([
        {"hash": h1, "kind": "image", "inline": True},      # eager → resolved
        {"hash": h2, "kind": "image", "inline": False},     # lazy image → skipped
        {"hash": h3, "kind": "document", "inline": True},   # document → skipped
    ])
    assert out == [b"IMGBYTES"]
    agent.storage.retrieve_file.assert_awaited_once_with(h1)


@pytest.mark.asyncio
async def test_resolve_eager_images_skips_missing_bytes():
    from kestrel_sovereign.agent.streaming import StreamingMixin
    agent = MagicMock()
    agent.storage.retrieve_file = AsyncMock(return_value=None)  # bytes gone
    resolve = StreamingMixin._resolve_eager_images.__get__(agent)
    out = await resolve([{"hash": "a" * 64, "kind": "image", "inline": True}])
    assert out == []


@pytest.mark.asyncio
async def test_resolve_eager_images_empty_and_no_storage():
    from kestrel_sovereign.agent.streaming import StreamingMixin
    agent = MagicMock()
    resolve = StreamingMixin._resolve_eager_images.__get__(agent)
    assert await resolve(None) == []
    assert await resolve([]) == []
    # An agent with no storage facade resolves to nothing (no crash).
    agent2 = MagicMock(spec=[])
    resolve2 = StreamingMixin._resolve_eager_images.__get__(agent2)
    assert await resolve2(
        [{"hash": "a" * 64, "kind": "image", "inline": True}]) == []


# --- sanitizer inline flag --------------------------------------------------

def test_sanitize_attachments_inline_flag():
    from kestrel_sovereign.endpoints.agent import _sanitize_attachments
    h = "d" * 64
    out = _sanitize_attachments([
        {"hash": h, "kind": "image", "mime": "image/png", "inline": True},
        {"hash": h, "kind": "image", "inline": False},
        {"hash": h, "kind": "document", "inline": True},   # docs can't ride inline
    ])
    assert out[0]["inline"] is True
    assert out[1]["inline"] is False
    assert out[2]["inline"] is False
