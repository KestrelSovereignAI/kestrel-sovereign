"""Adapter-level tests for the ``contribute_system_prompt`` hook (#807 / #806).

Verifies that:
- ``LLMAdapter.contribute_system_prompt`` defaults to identity.
- ``CodexAdapter`` and ``OpenAIAdapter`` apply the GPT-5 overlay.
- ``AnthropicAdapter`` (and any other adapter) inherits the identity default.
- ``_apply_system_prompt_contribution`` correctly augments the first system
  message in a chat-completions-style list, leaves other roles alone, and
  prepends a system message when none exists.
"""

import pytest

from kestrel_sovereign.llm.adapter import LLMAdapter
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter


class _BareAdapter(LLMAdapter):
    """Concrete bare-bones adapter for testing the base-class default."""

    async def get_response(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


class TestBaseHookIdentity:
    def test_default_returns_base_unchanged(self):
        adapter = _BareAdapter()
        assert adapter.contribute_system_prompt("gpt-5.4", "hello") == "hello"
        # Even for a model the overrides would normally augment, the base is
        # identity — no behavior leaks from subclass implementations into the
        # base hook itself.
        assert adapter.contribute_system_prompt("gpt-4o", None) is None

    def test_anthropic_adapter_inherits_identity(self):
        # Avoid importing AnthropicAdapter at module scope to keep this test
        # independent of the optional anthropic SDK.
        from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()
        assert adapter.contribute_system_prompt("gpt-5.4", "hello") == "hello"
        assert adapter.contribute_system_prompt("claude-sonnet-4-5", "x") == "x"


class TestCodexAdapterOverlay:
    def test_applies_overlay_for_gpt5(self):
        adapter = CodexAdapter()
        out = adapter.contribute_system_prompt("gpt-5.4", "You are Kestrel.")
        assert out is not None
        assert out.startswith("<persona_latch>")
        assert out.endswith("You are Kestrel.")

    def test_skips_overlay_for_non_gpt5(self):
        adapter = CodexAdapter()
        assert adapter.contribute_system_prompt("gpt-4o", "x") == "x"
        assert adapter.contribute_system_prompt("o1-preview", "y") == "y"

    @pytest.mark.parametrize(
        "model_id",
        ["gpt-5", "gpt-5.4", "gpt-5.4-codex", "gpt-5.5-pro", "gpt-5.4-mini"],
    )
    def test_overlay_applies_across_gpt5_variants(self, model_id):
        adapter = CodexAdapter()
        out = adapter.contribute_system_prompt(model_id, "base")
        assert "<execution_policy>" in out
        assert "<tool_discipline>" in out


class TestOpenAIAdapterOverlay:
    def test_applies_overlay_for_gpt5(self):
        adapter = OpenAIAdapter()
        out = adapter.contribute_system_prompt("gpt-5.4", "constitutional")
        assert out is not None
        assert "<persona_latch>" in out
        assert "constitutional" in out

    def test_skips_overlay_for_non_gpt5(self):
        adapter = OpenAIAdapter()
        assert adapter.contribute_system_prompt("gpt-4o", "x") == "x"


class TestApplySystemPromptContributionToMessages:
    def test_augments_first_system_message_only(self):
        adapter = OpenAIAdapter()
        messages = [
            {"role": "system", "content": "You are Kestrel."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "ignored second system"},
        ]
        out = adapter._apply_system_prompt_contribution(messages, "gpt-5.4")
        assert out[0]["content"].startswith("<persona_latch>")
        assert out[0]["content"].endswith("You are Kestrel.")
        assert out[1] == {"role": "user", "content": "hi"}
        assert out[2] == {"role": "system", "content": "ignored second system"}

    def test_unchanged_when_no_contribution(self):
        adapter = OpenAIAdapter()
        messages = [
            {"role": "system", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        out = adapter._apply_system_prompt_contribution(messages, "gpt-4o")
        assert out == messages

    def test_does_not_mutate_input(self):
        adapter = OpenAIAdapter()
        messages = [{"role": "system", "content": "base"}]
        original = [dict(m) for m in messages]
        _ = adapter._apply_system_prompt_contribution(messages, "gpt-5.4")
        assert messages == original

    def test_prepends_system_when_none_exists(self):
        adapter = OpenAIAdapter()
        messages = [{"role": "user", "content": "hi"}]
        out = adapter._apply_system_prompt_contribution(messages, "gpt-5.4")
        assert out[0]["role"] == "system"
        assert out[0]["content"].startswith("<persona_latch>")
        assert out[1] == {"role": "user", "content": "hi"}

    def test_preserves_messages_when_empty_and_no_contribution(self):
        adapter = OpenAIAdapter()
        out = adapter._apply_system_prompt_contribution([], "gpt-4o")
        assert out == []

    def test_skips_non_string_system_content_silently(self):
        # Multi-part content (e.g. tool-call recap) is left alone — the overlay
        # is intended for ordinary text system prompts.
        adapter = OpenAIAdapter()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "base"}]},
            {"role": "user", "content": "x"},
        ]
        out = adapter._apply_system_prompt_contribution(messages, "gpt-5.4")
        # Existing system message untouched; new system prepended for the contribution.
        assert out[0]["role"] == "system"
        assert out[0]["content"].startswith("<persona_latch>")
        assert out[1]["content"] == [{"type": "text", "text": "base"}]


class TestCodexInstructionsAugmentation:
    """Verify the codex flow: extract → contribute → sent to the app-server.

    The app-server-backed adapter passes the (overlaid) system prompt as
    ``developerInstructions`` on ``thread/start``. These assert the same
    extract→contribute→wire chain the old ``_build_request_body`` test
    covered, just at the new boundary.
    """

    class _CaptureApp:
        def __init__(self):
            self.thread_start_params = None

        async def ensure_started(self):
            pass

        async def request(self, method, params=None, *, timeout=120):
            if method == "thread/start":
                self.thread_start_params = params
                return {"thread": {"id": "t1"}}
            return {}

    @pytest.mark.asyncio
    async def test_instructions_carry_overlay_for_gpt5(self):
        from kestrel_sovereign.llm.codex_adapter import (
            _extract_instructions_and_input,
        )

        adapter = CodexAdapter()
        instructions, _ = _extract_instructions_and_input(
            [{"role": "system", "content": "You are Kestrel."},
             {"role": "user", "content": "hi"}]
        )
        instructions = adapter.contribute_system_prompt("gpt-5.4", instructions)
        app = self._CaptureApp()
        await adapter._ensure_thread(app, "s", "gpt-5.4", instructions, None)
        sent = app.thread_start_params["developerInstructions"]
        assert sent.startswith("<persona_latch>")
        assert sent.endswith("You are Kestrel.")

    @pytest.mark.asyncio
    async def test_instructions_unchanged_for_non_gpt5(self):
        from kestrel_sovereign.llm.codex_adapter import (
            _extract_instructions_and_input,
        )

        adapter = CodexAdapter()
        instructions, _ = _extract_instructions_and_input(
            [{"role": "system", "content": "You are Kestrel."},
             {"role": "user", "content": "hi"}]
        )
        instructions = adapter.contribute_system_prompt("gpt-4o", instructions)
        app = self._CaptureApp()
        await adapter._ensure_thread(app, "s", "gpt-4o", instructions, None)
        assert app.thread_start_params["developerInstructions"] == "You are Kestrel."
