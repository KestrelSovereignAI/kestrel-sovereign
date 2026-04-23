"""Tests for the dynamic_user_context field on ContextResult (issue #703).

Purpose: verify that memory and RAG content is placed in the new
`dynamic_user_context` string rather than concatenated into `system_prompt`,
so the system prefix remains stable across turns and downstream LLM prompt
caches (llama.cpp per-slot KV, OpenAI prefix cache, Anthropic cache_control)
can actually hit.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.context_manager import ContextManager, ContextResult


def _make_cm(
    *,
    memories_result: str = "",
    rag_result: str = "",
    history: list | None = None,
) -> ContextManager:
    """Build a ContextManager with all dependencies mocked.

    Token counts are stubbed to len(text)//4 (rough approximation) so budget
    accounting still functions; the tests care about string composition, not
    token math.
    """
    cm = object.__new__(ContextManager)
    cm.MICROCOMPACT_KEEP_RECENT = 5
    cm.EPISODE_THRESHOLD_MESSAGES = 20
    cm.agent_id = "test-agent"
    cm.storage = MagicMock()

    # Minimal token counter
    counter = MagicMock()
    counter.count = lambda s: max(1, len(s) // 4) if s else 0
    counter.count_messages = lambda msgs: sum(
        (len(m.get("content", "")) // 4) + 4 for m in msgs
    )
    cm._counter = counter
    cm._counter_model = "test-model"

    # No llm_service → state_of_mind/prompt_adaptation path skipped
    cm._llm_service = None
    cm.llm_service = None
    cm._model_fallback = "test-model"

    # Conversation history comes pre-supplied
    cm.conversation_manager = MagicMock()
    cm.conversation_manager.get_conversation_history = AsyncMock(
        return_value=history or []
    )

    # Memory retriever: if memories_result non-empty, retrieval returns it
    cm.memory_retriever = MagicMock() if memories_result else None
    cm.memory_manager = MagicMock()
    cm.memory_manager.retrieve_memories = AsyncMock(return_value=memories_result)

    # RAG via context_builder
    cm.context_builder = MagicMock()
    cm.context_builder.build_system_prompt = MagicMock(
        return_value="SYSTEM_PROMPT_BASE"
    )
    cm.context_builder.retrieve_context = AsyncMock(return_value=rag_result)
    cm.context_builder.format_conversation_history = MagicMock(
        side_effect=lambda history, max_tokens: list(history)
    )
    cm.context_builder.get_episode_context = AsyncMock(return_value="")

    cm.consolidator = None
    return cm


@pytest.mark.asyncio
async def test_system_prompt_stable_when_memory_and_rag_differ():
    """Two consecutive turns with different retrieved memories + different RAG
    must produce byte-identical `system_prompt`.  This is the core cache
    invariant the ticket was filed to enforce.
    """
    cm_turn_1 = _make_cm(
        memories_result="[Memory 1] First fact", rag_result="[Document A] Content A"
    )
    result_1 = await cm_turn_1.build_context(
        query="what's the weather?",
        constitution="CONSTITUTION_TEXT",
    )

    cm_turn_2 = _make_cm(
        memories_result="[Memory 2] Entirely different fact",
        rag_result="[Document B] Different content",
    )
    result_2 = await cm_turn_2.build_context(
        query="who are you?",
        constitution="CONSTITUTION_TEXT",
    )

    assert result_1.system_prompt == result_2.system_prompt, (
        "system_prompt MUST be byte-identical across turns for prompt cache "
        "to hit. Divergence means memory/RAG leaked back into the system "
        "message."
    )


@pytest.mark.asyncio
async def test_dynamic_user_context_wraps_memories_and_rag():
    """When both memories and RAG return content, dynamic_user_context must
    contain both blocks, each under its own XML tag, inside a single outer
    <retrieved_context> wrapper.
    """
    cm = _make_cm(
        memories_result="[Memory 1] test memory",
        rag_result="[Document] test document",
    )
    result = await cm.build_context(
        query="test query",
        constitution="CONSTITUTION_TEXT",
    )

    assert result.dynamic_user_context.startswith("<retrieved_context>")
    assert result.dynamic_user_context.endswith("</retrieved_context>")
    assert "<memories>" in result.dynamic_user_context
    assert "</memories>" in result.dynamic_user_context
    assert "[Memory 1] test memory" in result.dynamic_user_context
    assert "<documents>" in result.dynamic_user_context
    assert "</documents>" in result.dynamic_user_context
    assert "[Document] test document" in result.dynamic_user_context


@pytest.mark.asyncio
async def test_dynamic_user_context_empty_when_no_retrieval():
    """No memory hits and no RAG hits → dynamic_user_context must be the empty
    string, NOT a dangling <retrieved_context></retrieved_context> shell
    (which would waste tokens and confuse the model on a short/first turn).
    """
    cm = _make_cm(memories_result="", rag_result="")
    result = await cm.build_context(
        query="first turn",
        constitution="CONSTITUTION_TEXT",
    )

    assert result.dynamic_user_context == ""


@pytest.mark.asyncio
async def test_dynamic_user_context_memories_only_when_rag_empty():
    """Memories retrieved but RAG empty → only the memories block appears,
    still inside the <retrieved_context> wrapper.  No empty <documents/> tag.
    """
    cm = _make_cm(memories_result="[Memory] hit", rag_result="")
    result = await cm.build_context(
        query="q",
        constitution="CONSTITUTION_TEXT",
    )

    assert "<retrieved_context>" in result.dynamic_user_context
    assert "<memories>" in result.dynamic_user_context
    assert "<documents>" not in result.dynamic_user_context


@pytest.mark.asyncio
async def test_dynamic_user_context_rag_only_when_memories_empty():
    """RAG retrieved but no memory hits → only the documents block appears."""
    cm = _make_cm(memories_result="", rag_result="[Document] hit")
    result = await cm.build_context(
        query="q",
        constitution="CONSTITUTION_TEXT",
    )

    assert "<retrieved_context>" in result.dynamic_user_context
    assert "<documents>" in result.dynamic_user_context
    assert "<memories>" not in result.dynamic_user_context


@pytest.mark.asyncio
async def test_system_prompt_does_not_contain_memory_or_rag_markers():
    """Belt-and-suspenders: the raw memory string and the RAG-document string
    must NOT appear inside system_prompt.  If they do, the ticket's fix has
    regressed and the cache is broken again.
    """
    cm = _make_cm(
        memories_result="[Memory] secret fact",
        rag_result="[Document] secret doc",
    )
    result = await cm.build_context(
        query="q",
        constitution="CONSTITUTION_TEXT",
    )

    assert "[Memory] secret fact" not in result.system_prompt
    assert "[Document] secret doc" not in result.system_prompt
    assert "RELEVANT DOCUMENTS" not in result.system_prompt


@pytest.mark.asyncio
async def test_reflection_guidance_stays_in_system_not_dynamic():
    """Reflection guidance is conversation-stable and belongs in the cacheable
    system prefix.  It must NOT leak into dynamic_user_context.
    """
    cm = _make_cm(memories_result="", rag_result="")
    result = await cm.build_context(
        query="q",
        constitution="CONSTITUTION_TEXT",
        reflection_guidance=["always be kind", "use tools when asked"],
    )

    assert "ACTIVE REFLECTION GUIDANCE" in result.system_prompt
    assert "always be kind" in result.system_prompt
    assert "always be kind" not in result.dynamic_user_context
    assert result.dynamic_user_context == ""


@pytest.mark.asyncio
async def test_ephemeral_mode_dynamic_is_empty():
    """EPHEMERAL privacy mode does no retrieval, so dynamic_user_context must
    be empty regardless of what the mocks would otherwise return.
    """
    cm = _make_cm(memories_result="[M] x", rag_result="[D] y")
    result = await cm.build_context(
        query="q",
        constitution="CONSTITUTION_TEXT",
        privacy_mode="EPHEMERAL",
    )

    assert result.dynamic_user_context == ""


@pytest.mark.asyncio
async def test_context_result_default_dynamic_user_context_is_empty_string():
    """Sanity: the dataclass default is the empty string (NOT None), so
    downstream format() calls never see None.
    """
    r = ContextResult(
        system_prompt="x", messages=[], total_tokens=0, budget_summary={}
    )
    assert r.dynamic_user_context == ""


def test_user_prompt_template_substitutes_empty_dynamic_context():
    """user_prompt.md must survive a format() with an empty `context` —
    first-turn conversations legitimately have nothing retrieved yet.
    """
    from pathlib import Path
    import re

    template_path = (
        Path(__file__).parent.parent.parent
        / "kestrel_sovereign/prompts/user_prompt.md"
    )
    raw = template_path.read_text()
    match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    assert match, "user_prompt.md must contain a fenced template block"
    template = match.group(1).strip()

    # Empty context must not raise and must still carry the query.
    rendered = template.format(context="", query="hello")
    assert "hello" in rendered
    assert "{context}" not in rendered  # placeholder fully substituted


def test_user_prompt_template_substitutes_populated_dynamic_context():
    """When the dynamic context is populated (memories + RAG), it must
    appear BEFORE the wrapped query in the rendered output.  The framing
    scaffolding that used to live in this template has moved to the system
    prompt (issue #703) so the template itself is intentionally minimal.
    """
    from pathlib import Path
    import re

    template_path = (
        Path(__file__).parent.parent.parent
        / "kestrel_sovereign/prompts/user_prompt.md"
    )
    raw = template_path.read_text()
    match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    template = match.group(1).strip()

    ctx = (
        "<retrieved_context>\n"
        "<memories>\n[Memory] alpha\n</memories>\n"
        "<documents>\n[Document] beta\n</documents>\n"
        "</retrieved_context>"
    )
    rendered = template.format(context=ctx, query="hello")

    # Query survives and retrieved content is present in expected order.
    context_start = rendered.index("<retrieved_context>")
    query_start = rendered.index("hello")
    assert context_start < query_start, (
        "retrieved context must appear BEFORE the query so the model reads "
        "it as priming for the answer"
    )
    assert "[Memory] alpha" in rendered
    assert "[Document] beta" in rendered
