"""Integration test: Anthropic cache_control works end-to-end (issue #705).

Requires ``ANTHROPIC_API_KEY`` in env (or ``.env`` — auto-loaded via
python-dotenv).  Skipped in CI unless the key is present; useful
locally to confirm the cache hints are actually landing and the server
reports cache_read_input_tokens on turn 2.

The test intentionally uses a system prompt large enough (~1200+
tokens) to exceed Anthropic's 1024-token minimum cache size for Opus
and Sonnet.  If the model under test is a Haiku variant, the cache
won't kick in below 2048 tokens — use a different model for this test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Auto-load .env so running this file directly picks up ANTHROPIC_API_KEY.
try:
    from dotenv import load_dotenv as _load_dotenv
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        _load_dotenv(_env, override=False)
except ImportError:
    pass

import anthropic  # noqa: E402
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter  # noqa: E402


PADDING = "\n".join(
    f"- Reminder {i}: be helpful and consistent across turns."
    for i in range(200)
)

LARGE_SYSTEM_PROMPT = (
    "You are a benchmark assistant for issue #705.  Reply in one short "
    "sentence, nothing else.  Additional reminders follow so the system "
    "prompt exceeds Anthropic's minimum cacheable prefix length:\n"
    f"{PADDING}"
)


def _api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def _default_model() -> str:
    # Pin to Sonnet (1024-token floor) so the LARGE_SYSTEM_PROMPT exceeds
    # the minimum.  Caller can override via env.
    return os.environ.get(
        "ANTHROPIC_BENCH_MODEL", "claude-sonnet-4-5-20250929"
    )


@pytest.mark.asyncio
async def test_anthropic_cache_read_on_turn_two_real_api():
    """Real two-turn conversation against Anthropic.  Asserts the
    end-to-end cache story holds:

      * Cross-turn caching WORKS — turn 2 reads the system prefix from
        cache (non-zero `cache_read_input_tokens`).  This is the whole
        point of this ticket.
      * Uncached ("regular") input on turn 2 is only the new tail
        content: the fresh user question plus the newly-cached previous-
        turn delta.  Specifically, `input_tokens` on turn 2 should be
        much smaller than turn 1's system prompt size.

    Deliberately does NOT assert on turn 1's cache_creation, because
    Anthropic's cache may already hold this system prompt from prior
    runs of this test — in that case turn 1 reads and turn 2 also reads.
    The invariant we care about is "the same prefix hits cache across
    turns", and that's covered by the T2 read assertion.
    """
    key = _api_key()
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping real-network test")

    client = anthropic.AsyncAnthropic(api_key=key)
    adapter = AnthropicAdapter()
    model = _default_model()

    # Turn 1: first message in the conversation.
    messages_t1 = [
        {"role": "system", "content": LARGE_SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2+2?  Numeric answer only."},
    ]
    resp_t1 = await adapter.get_response(
        client=client,
        model=model,
        messages=messages_t1,
        max_tokens=16,
    )
    raw_t1 = resp_t1.raw
    usage_t1 = getattr(raw_t1, "usage", None)
    cache_creation_t1 = getattr(usage_t1, "cache_creation_input_tokens", 0) or 0
    cache_read_t1 = getattr(usage_t1, "cache_read_input_tokens", 0) or 0
    input_tokens_t1 = getattr(usage_t1, "input_tokens", 0) or 0

    # Turn 2: same system prompt + a new user question, plus turn 1's
    # exchange in history (penultimate marker caches that exchange for
    # later turns).  The system prefix should hit cache here.
    messages_t2 = messages_t1 + [
        {"role": "assistant", "content": resp_t1.content or ""},
        {"role": "user", "content": "What is 3+3?  Numeric answer only."},
    ]
    resp_t2 = await adapter.get_response(
        client=client,
        model=model,
        messages=messages_t2,
        max_tokens=16,
    )
    raw_t2 = resp_t2.raw
    usage_t2 = getattr(raw_t2, "usage", None)
    cache_creation_t2 = getattr(usage_t2, "cache_creation_input_tokens", 0) or 0
    cache_read_t2 = getattr(usage_t2, "cache_read_input_tokens", 0) or 0
    input_tokens_t2 = getattr(usage_t2, "input_tokens", 0) or 0

    try:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
    except Exception:
        pass

    # Report to stderr for local-run visibility.
    print(
        f"\n[{model}] "
        f"T1 input={input_tokens_t1} write={cache_creation_t1} read={cache_read_t1} | "
        f"T2 input={input_tokens_t2} write={cache_creation_t2} read={cache_read_t2}",
        file=sys.stderr,
    )

    # Core invariant: cross-turn caching works.  On turn 2 the system
    # prefix is read from cache.
    assert cache_read_t2 > 0, (
        "Turn 2 should have read from cache (non-zero "
        "cache_read_input_tokens).  If zero, the marker placement is not "
        "producing a stable prefix across turns.\n"
        f"T1 write={cache_creation_t1} read={cache_read_t1} input={input_tokens_t1}\n"
        f"T2 write={cache_creation_t2} read={cache_read_t2} input={input_tokens_t2}"
    )

    # Corollary: uncached tokens on turn 2 should be much smaller than
    # the system prompt (which is now cached).  LARGE_SYSTEM_PROMPT is
    # ~2000 tokens; turn 2's `input_tokens` should be well under that.
    # (It covers only the new user message — the history marker covers
    # turn 1's user/assistant exchange.)
    assert input_tokens_t2 < 200, (
        "Turn 2's uncached input_tokens is larger than expected — the "
        "cache is missing more than the new user message.\n"
        f"input_tokens_t2={input_tokens_t2}"
    )


@pytest.mark.asyncio
async def test_anthropic_tiny_prompt_silent_no_op_under_threshold():
    """A sub-threshold prompt (< 1024 tokens total) should still succeed —
    Anthropic silently no-ops the cache_control markers rather than
    erroring.  The response reports zero cache_read and zero
    cache_creation.  Confirms the adapter doesn't break on short prompts.
    """
    key = _api_key()
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping real-network test")

    client = anthropic.AsyncAnthropic(api_key=key)
    adapter = AnthropicAdapter()
    model = _default_model()

    # Intentionally tiny — well below 1024 tokens.
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Reply 'ok' only."},
    ]

    resp = await adapter.get_response(
        client=client, model=model, messages=messages, max_tokens=8
    )
    try:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
    except Exception:
        pass

    usage = getattr(resp.raw, "usage", None)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # Silent no-op: both should be zero (or absent — None).  Non-zero on
    # the first call of a sub-threshold prompt would mean Anthropic
    # started accepting below-threshold caches, which would be a nice
    # upgrade but nothing here depends on.
    assert cache_read == 0, (
        f"Expected zero cache_read on sub-threshold prompt, got {cache_read}"
    )
    # We do NOT assert cache_creation == 0 — Anthropic may decide to
    # cache anyway in the future, and that would be fine.
