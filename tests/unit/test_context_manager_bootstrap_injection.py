"""Regression test: system prompts sent via ContextManager actually
contain the agent's bootstrap identity (SOUL.md).

Before this fix, ContextManager created its own internal ContextBuilder
without an ``agent_data_path``, so its BootstrapLoader short-circuited
and never loaded SOUL.md.  KestrelAgent had a SEPARATE, correctly-
configured ContextBuilder, but that one was unused by the chat path.
Net effect: every chat turn went to the LLM with no identity block,
and the agent couldn't answer "what is your name?" even though SOUL.md
sat right there on disk saying 'You are Nellie.'

The fix is dependency injection — KestrelAgent passes its ContextBuilder
(with bootstrap already loaded) into ContextManager, which uses it
instead of creating a second one.  These tests pin that contract so the
bug can't silently regress.
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.agent.context_manager import ContextManager


def _stub_context_builder_with_soul(soul_text: str) -> ContextBuilder:
    """Build a minimal-but-real ContextBuilder whose BootstrapLoader has
    SOUL.md preloaded.  Mirrors what KestrelAgent does in its __init__:
    construct the ContextBuilder with an agent_data_path that contains
    SOUL.md, and BootstrapLoader.load() picks it up.
    """
    cb = object.__new__(ContextBuilder)
    # Populate the fields the @property accessors read from BEFORE any
    # property is touched — order matters because build_system_prompt
    # will invoke self.counter, which reads self.model, which reads
    # self._llm_service.
    cb._llm_service = None
    cb._model_fallback = "test-model"
    fake_counter = MagicMock()
    fake_counter.count = lambda s: max(1, len(s) // 4) if s else 0
    fake_counter.count_messages = lambda msgs: sum(
        (len(m.get("content", "")) // 4) + 4 for m in msgs
    )
    cb._counter = fake_counter
    cb._counter_model = "test-model"

    # BootstrapLoader stub that returns SOUL.md as the one bootstrap file.
    cb._bootstrap_loader = MagicMock()
    cb._bootstrap_loader.load = MagicMock(
        return_value={"SOUL.md": soul_text}
    )
    cb._bootstrap_loader.get_file = MagicMock(
        side_effect=lambda name: soul_text if name == "SOUL.md" else None
    )

    cb.storage = MagicMock()
    cb.get_session_briefing = lambda: ""
    cb.get_episode_context = AsyncMock(return_value="")
    cb.retrieve_context = AsyncMock(return_value="")
    cb.format_conversation_history = MagicMock(
        side_effect=lambda history, **kw: list(history)
    )
    return cb


def test_context_manager_uses_injected_context_builder():
    """When a ContextBuilder is passed to ContextManager, that exact
    instance is what ContextManager uses — no silent second one.  This
    is the load-bearing invariant: if ContextManager were ever to
    rebuild its own internally, bootstrap would be lost again.
    """
    injected = _stub_context_builder_with_soul("You are Nellie.")
    cm = ContextManager(
        storage=MagicMock(),
        context_builder=injected,
    )
    assert cm.context_builder is injected, (
        "ContextManager must reuse the injected ContextBuilder instance, "
        "not wrap or copy it — otherwise mutations by the bootstrap "
        "feature (e.g. !reload-soul) stop propagating."
    )


def test_context_manager_falls_back_when_no_injection():
    """For test harnesses and callers that don't have an agent-owned
    ContextBuilder handy, ContextManager still constructs a sensible
    fallback.  Bootstrap content will be empty in that mode — by design,
    because there's no agent_data_path to scan.
    """
    cm = ContextManager(storage=MagicMock())
    assert cm.context_builder is not None
    # Fallback path builds a fresh ContextBuilder with no bootstrap cache.
    # Spot-check: it's a real ContextBuilder (not a subclass stub).
    assert isinstance(cm.context_builder, ContextBuilder)


@pytest.mark.asyncio
async def test_system_prompt_contains_soul_content_when_injected():
    """End-to-end invariant: when ContextManager.build_context() runs on
    a real conversation turn, the returned system_prompt includes the
    agent's SOUL.md content verbatim (wrapped in the YOUR IDENTITY
    boundary markers).

    This is the actual fix signal — before the fix, the returned
    system_prompt had no identity block regardless of what SOUL.md
    contained; after the fix, the injected ContextBuilder surfaces it.
    """
    soul_text = (
        "# SOUL.md - You Are Nellie\n"
        "You are Nellie, the skeptical Kestrel.\n"
    )
    injected = _stub_context_builder_with_soul(soul_text)

    cm = object.__new__(ContextManager)
    cm.MICROCOMPACT_KEEP_RECENT = 5
    cm.EPISODE_THRESHOLD_MESSAGES = 20
    cm.agent_id = "nellie"
    cm.storage = MagicMock()
    cm._llm_service = None
    cm.llm_service = None
    cm._model_fallback = "test-model"
    cm._counter = injected.counter
    cm._counter_model = "test-model"
    cm.context_builder = injected
    cm.conversation_manager = MagicMock()
    cm.memory_manager = MagicMock()
    cm.memory_manager.retrieve_memories = AsyncMock(return_value="")
    cm.memory_retriever = None
    cm.consolidator = None

    result = await cm.build_context(
        query="who are you?",
        constitution="CONSTITUTION",
        conversation_history=[],
    )
    assert "You are Nellie, the skeptical Kestrel." in result.system_prompt, (
        "system_prompt must contain SOUL.md content — the agent relies on "
        "this to know its own identity.  If this assertion fails the "
        "ContextBuilder's BootstrapLoader is empty, which means the "
        "injection contract is broken."
    )
    assert "YOUR IDENTITY" in result.system_prompt, (
        "SOUL.md must be wrapped in the YOUR IDENTITY boundary markers "
        "so the model treats it as identity, not arbitrary context."
    )


def test_kestrel_agent_injects_its_own_context_builder():
    """KestrelAgent pre-loads SOUL.md into its own ContextBuilder during
    __init__; ContextManager must receive THAT exact builder.  Grepping
    the source is the surest way to pin this without standing up a full
    KestrelAgent — the one-liner either exists or it doesn't.
    """
    import inspect
    from kestrel_sovereign import kestrel_agent

    source = inspect.getsource(kestrel_agent)
    # Must contain both the ContextManager instantiation AND the
    # injection kwarg on that call.  Checking the substring order so
    # anyone moving the instantiation keeps the injection with it.
    cm_idx = source.find("self.context_manager = ContextManager(")
    assert cm_idx != -1, "ContextManager instantiation vanished from KestrelAgent"
    # Next close-paren after that call; inject must appear between.
    close_idx = source.index(")", cm_idx)
    call_block = source[cm_idx:close_idx]
    assert "context_builder=self.context_builder" in call_block, (
        "KestrelAgent must inject its own context_builder into "
        "ContextManager so bootstrap files (SOUL.md) flow through."
    )
