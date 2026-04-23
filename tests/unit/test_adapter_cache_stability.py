"""Per-adapter cache-stability tests.

Each provider adapter transforms Kestrel's canonical OpenAI-shaped
`messages[]` into its own provider's request format (OpenAI SDK, Anthropic
SDK, Ollama HTTP, Gemini, etc.).  If an adapter's transformation is
non-deterministic — reorders, adds timestamps, re-serializes in a way that
differs across turns — the token-level prefix the provider sees will
diverge even though Kestrel's input was stable, and prompt caching won't
hit.

These tests run each adapter with two consecutive "turns" of realistic
messages: identical system + shared history + different final user turn.
They intercept the adapter's SDK call and assert that the stable-prefix
portions of the emitted request are byte-identical across the two turns.
No network is touched.
"""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter


# ---------------------------------------------------------------------------
# Shared fixtures: representative turn-1 and turn-2 message arrays that a
# real Kestrel request would produce after issue #703.  The system prompt
# contains no per-turn content (no memory / RAG); the first history user
# turn is wrapped; the current-turn user carries the per-turn retrieved
# context block.
# ---------------------------------------------------------------------------

STABLE_SYSTEM = (
    "--- GOVERNING CONSTITUTION ---\n"
    "A stable constitution that does not vary between turns.\n"
    "--- END CONSTITUTION ---\n"
    "--- INPUT SECURITY & TAGGED INPUT ---\n"
    "(anti-injection block — same every turn)\n"
    "--- END ---"
)


def _turn_messages(
    *, history: List[Dict[str, Any]], retrieved: str, query: str
) -> List[Dict[str, Any]]:
    """Assemble the messages list Kestrel would send for one turn after
    issue #703's restructure.  Stable system, append-only history with
    user messages already wrapped, and a current-turn user message whose
    content is `<retrieved_context>...</retrieved_context>\n<user_input>q</user_input>`.
    """
    current = (
        f"<retrieved_context>\n{retrieved}\n</retrieved_context>\n"
        f"<user_input>\n{query}\n</user_input>"
        if retrieved
        else f"<user_input>\n{query}\n</user_input>"
    )
    return (
        [{"role": "system", "content": STABLE_SYSTEM}]
        + history
        + [{"role": "user", "content": current}]
    )


def _prior_history_and_new_history():
    """Simulate the append-only growth of history between turns 2 and 3
    of a conversation.  `history_t2` is what turn 2 would send as history
    (turn 1's exchange).  `history_t3` is what turn 3 would send (turns 1
    and 2's exchanges).  User turns are pre-wrapped because that's what
    `format_conversation_history` produces post-#703.
    """
    history_t2 = [
        {"role": "user", "content": "<user_input>\nq1\n</user_input>"},
        {"role": "assistant", "content": "a1"},
    ]
    history_t3 = history_t2 + [
        {"role": "user", "content": "<user_input>\nq2\n</user_input>"},
        {"role": "assistant", "content": "a2"},
    ]
    return history_t2, history_t3


# ---------------------------------------------------------------------------
# Adapter-specific interception helpers.  Each wires a fake SDK client and
# runs the adapter's get_response, returning the args the adapter passed
# to the SDK.
# ---------------------------------------------------------------------------


async def _run_openai_and_capture(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Drive the OpenAI adapter; return the kwargs sent to `chat.completions.create`."""
    fake_client = MagicMock()
    # chat.completions.create is awaited by the adapter.
    create_call = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )
    )
    fake_client.chat.completions.create = create_call

    adapter = OpenAIAdapter()
    await adapter.get_response(
        client=fake_client,
        model="gpt-4o-mini",
        messages=messages,
    )
    captured = create_call.call_args.kwargs
    return captured


async def _run_anthropic_and_capture(
    messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Drive the Anthropic adapter; return kwargs sent to `messages.create`."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=1),
        )
    )

    adapter = AnthropicAdapter()
    await adapter.get_response(
        client=fake_client,
        model="claude-sonnet-4-5-20250929",
        messages=messages,
    )
    captured = fake_client.messages.create.call_args.kwargs
    return captured


async def _run_ollama_and_capture(
    messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Drive the Ollama adapter; return kwargs sent to `client.chat`."""
    fake_client = MagicMock()
    chat_call = AsyncMock(
        return_value={
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 1,
        }
    )
    fake_client.chat = chat_call

    adapter = OllamaAdapter()
    await adapter.get_response(
        client=fake_client,
        model="llama3.2:3b",
        messages=messages,
    )
    # Ollama adapter invokes client.chat via with_retry; captured kwargs
    # include `messages=` verbatim.
    return {"json": {"messages": chat_call.call_args.kwargs["messages"]}}


# ---------------------------------------------------------------------------
# Per-adapter parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,label",
    [
        (_run_openai_and_capture, "openai"),
        (_run_anthropic_and_capture, "anthropic"),
        (_run_ollama_and_capture, "ollama"),
    ],
)
async def test_adapter_preserves_system_stability_across_turns(runner, label):
    """Across two consecutive turns with identical system and growing
    history, the adapter's outgoing payload must serialize the system
    identically on both turns.  For OpenAI/Ollama, that's messages[0].
    For Anthropic, it's the top-level `system` parameter.
    """
    history_t2, history_t3 = _prior_history_and_new_history()

    msgs_t2 = _turn_messages(
        history=history_t2, retrieved="[Memory] turn-2 mem", query="q2"
    )
    msgs_t3 = _turn_messages(
        history=history_t3, retrieved="[Memory] turn-3 mem", query="q3"
    )

    captured_t2 = await runner(msgs_t2)
    captured_t3 = await runner(msgs_t3)

    if label == "anthropic":
        # Anthropic extracts system into a top-level string; verify both
        # turns produce the byte-identical system.
        assert captured_t2.get("system") == captured_t3.get("system"), (
            f"[{label}] top-level system parameter diverged across turns"
        )
        assert captured_t2.get("system"), "[anthropic] system must be populated"
    elif label == "ollama":
        t2_msgs = captured_t2["json"]["messages"]
        t3_msgs = captured_t3["json"]["messages"]
        assert t2_msgs[0] == t3_msgs[0], (
            f"[{label}] first message (system) diverged across turns"
        )
    else:  # openai family
        t2_msgs = captured_t2["messages"]
        t3_msgs = captured_t3["messages"]
        assert t2_msgs[0] == t3_msgs[0], (
            f"[{label}] messages[0] (system) diverged across turns"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,label",
    [
        (_run_openai_and_capture, "openai"),
        (_run_ollama_and_capture, "ollama"),
    ],
)
async def test_openai_compatible_adapter_preserves_history_prefix(runner, label):
    """For OpenAI-shaped providers (OpenAI, OpenRouter, RunPod, llama.cpp
    via OpenAIAdapter; Ollama via its own adapter), the history messages
    present on turn 2 must appear byte-identically in turn 3's serialized
    payload at the same indices.  This is what prompt-cache longest-
    common-prefix matching requires.
    """
    history_t2, history_t3 = _prior_history_and_new_history()

    msgs_t2 = _turn_messages(
        history=history_t2, retrieved="", query="q2"
    )
    msgs_t3 = _turn_messages(
        history=history_t3, retrieved="", query="q3"
    )

    captured_t2 = await runner(msgs_t2)
    captured_t3 = await runner(msgs_t3)

    t2_msgs = (
        captured_t2["json"]["messages"] if label == "ollama" else captured_t2["messages"]
    )
    t3_msgs = (
        captured_t3["json"]["messages"] if label == "ollama" else captured_t3["messages"]
    )

    # For every index in msgs_t2 up to (but not including) the final user
    # turn, t3 must have the same content at that index.
    for i in range(len(t2_msgs) - 1):
        assert t2_msgs[i] == t3_msgs[i], (
            f"[{label}] payload diverged at messages[{i}] between turn 2 "
            f"and turn 3 — prompt cache would miss from this position "
            f"onward.\nt2: {t2_msgs[i]!r}\nt3: {t3_msgs[i]!r}"
        )


@pytest.mark.asyncio
async def test_anthropic_preserves_history_prefix_after_role_conversion():
    """Anthropic's adapter splits system out and keeps user/assistant in
    `messages`.  After that conversion, the leading user/assistant entries
    from turn 2 must byte-match the same positions in turn 3's converted
    messages (the Anthropic-format equivalent of the OpenAI-shaped history
    stability claim).
    """
    history_t2, history_t3 = _prior_history_and_new_history()
    msgs_t2 = _turn_messages(history=history_t2, retrieved="", query="q2")
    msgs_t3 = _turn_messages(history=history_t3, retrieved="", query="q3")

    captured_t2 = await _run_anthropic_and_capture(msgs_t2)
    captured_t3 = await _run_anthropic_and_capture(msgs_t3)

    anthropic_msgs_t2 = captured_t2["messages"]
    anthropic_msgs_t3 = captured_t3["messages"]

    for i in range(len(anthropic_msgs_t2) - 1):
        assert anthropic_msgs_t2[i] == anthropic_msgs_t3[i], (
            f"anthropic-format payload diverged at messages[{i}] between "
            f"turn 2 and turn 3.\nt2: {anthropic_msgs_t2[i]!r}\n"
            f"t3: {anthropic_msgs_t3[i]!r}"
        )


@pytest.mark.asyncio
async def test_openai_adapter_is_deterministic_on_identical_input():
    """Defensive: the OpenAI adapter must produce byte-identical serialized
    requests when called twice with exactly the same messages.  Any
    divergence (ordering, timestamping, randomized fields) would poison
    the cache even for truly repeated inputs.
    """
    history_t2, _ = _prior_history_and_new_history()
    msgs = _turn_messages(history=history_t2, retrieved="[M] m", query="q")
    cap_a = await _run_openai_and_capture(msgs)
    cap_b = await _run_openai_and_capture(msgs)
    # Serialize to canonical JSON for comparison (skip non-serializable bits
    # the mock may have injected — we only care about messages).
    a = json.dumps(cap_a.get("messages"), sort_keys=True)
    b = json.dumps(cap_b.get("messages"), sort_keys=True)
    assert a == b, "OpenAI adapter is non-deterministic on identical input"


@pytest.mark.asyncio
async def test_anthropic_adapter_is_deterministic_on_identical_input():
    """Same defensive check for the Anthropic adapter — system + converted
    messages must be stable on repeated calls with identical input.
    """
    history_t2, _ = _prior_history_and_new_history()
    msgs = _turn_messages(history=history_t2, retrieved="[M] m", query="q")
    cap_a = await _run_anthropic_and_capture(msgs)
    cap_b = await _run_anthropic_and_capture(msgs)
    sys_a = cap_a.get("system")
    sys_b = cap_b.get("system")
    msgs_a = json.dumps(cap_a.get("messages"), sort_keys=True)
    msgs_b = json.dumps(cap_b.get("messages"), sort_keys=True)
    assert sys_a == sys_b, "Anthropic adapter produced non-stable system"
    assert msgs_a == msgs_b, "Anthropic adapter produced non-stable messages"
