"""
Unified Context Manager for Kestrel Agent.

Orchestrates all context sources:
- Conversation history with token-aware truncation
- Episode summaries for long conversations
- Emotionally-weighted memory retrieval
- RAG document search (hybrid: embeddings + BM25)
- Constitutional grounding

This is the single integration point for context assembly,
replacing scattered context retrieval throughout the codebase.

Refactored to serve as an orchestration layer that delegates to specialized managers.
"""

import json
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

from .context_builder import ContextBuilder
from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, create_budget, DegradedModeError
from .conversation_manager import ConversationManager
from .memory_manager import MemoryManager
from .tool_context_manager import ToolContextManager

if TYPE_CHECKING:
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
    from kestrel_sovereign.storage.memory_retriever import MemoryRetriever
    from kestrel_sovereign.llm.service import LLMService

logger = logging.getLogger(__name__)


# Per-async-task tracking of constitutional-injection clauses for the
# current build_context call. The dispatcher reads this AFTER
# process_input returns so it can land in signal_log.injected_clauses_json
# / dropped_clauses_json. Using a ContextVar (not a shared agent
# attribute) so concurrent COGNITION dispatches don't race — each
# dispatch's task sees its OWN value (codex round-14 P2 catch).
_INJECTION_TRACKING_VAR: ContextVar[
    Optional[Tuple[Optional[List[str]], Optional[List[str]]]]
] = ContextVar("kestrel_constitution_injection_tracking", default=None)


def get_current_injection_tracking() -> Optional[
    Tuple[Optional[List[str]], Optional[List[str]]]
]:
    """Return the (injected, dropped) clause tuple for the CURRENT
    async task's most recent build_context invocation, or None.

    Used by `SignalDispatcher` after `agent.process_input` returns to
    populate signal_log audit fields. Per-task isolation via ContextVar
    means concurrent dispatches do not see each other's tracking.
    """
    return _INJECTION_TRACKING_VAR.get()


def reset_injection_tracking() -> None:
    """Clear the current task's injection tracking ContextVar.

    Called by `SignalDispatcher._dispatch_cognition` at dispatch
    start so an early-return `process_input` (safe mode, bootstrap,
    `!` command) doesn't leak the PREVIOUS turn's tracking into
    this dispatch's signal_log row (codex round-15 P2 fix).
    """
    _INJECTION_TRACKING_VAR.set(None)


@dataclass
class ContextResult:
    """Result of context assembly."""
    system_prompt: str
    messages: List[Dict[str, str]]
    total_tokens: int
    budget_summary: Dict[str, Any]
    episode_count: int = 0
    memory_count: int = 0
    rag_chunks: int = 0
    warnings: List[str] = field(default_factory=list)
    # Query-dependent context retrieved for this turn (memories + RAG).
    # Lives OUTSIDE the system message so the system prefix stays stable
    # across turns and downstream LLM prompt caches (llama.cpp per-slot KV,
    # OpenAI prefix cache, Anthropic cache_control) can actually hit.
    # Callers prepend this to the current user message content.
    dynamic_user_context: str = ""
    # Constitutional-injection tracking — populated only when the
    # caller supplies `system_prompt_budget_bytes` and the
    # priority-aware tracking assembler runs (kestrel-sovereign#1137).
    # The dispatcher reads these via `self._agent._last_injection_tracking`
    # after `process_input` returns and threads them into
    # `signal_log.injected_clauses_json` / `dropped_clauses_json`.
    injected_clauses: Optional[List[str]] = None
    dropped_clauses: Optional[List[str]] = None
    # Set to True when the elastic budget (#1309) raised
    # ``DegradedModeError`` because the measured mandatory governance
    # floor could not fit the model's context window. The caller MUST
    # surface this to the operator (UI, telemetry) instead of issuing
    # the LLM call under a false "normal" status — Emma's 2026-05-20
    # hardening invariant (Review record, PR #1306).
    degraded_mode: bool = False
    mandatory_system_tokens: int = 0


class ContextManager:
    """
    Unified context manager for the Kestrel agent.

    Responsibilities:
    1. Orchestrate specialized managers for different context concerns
    2. Manage token budgets across all context sources
    3. Decide when to use episodes vs raw history
    4. Coordinate RAG search with hybrid approach
    5. Ensure privacy mode compliance

    Usage:
        manager = ContextManager(storage)
        result = await manager.build_context(
            query="Tell me about our conversation yesterday",
            agent_id="agent-123",
            constitution=constitution_text
        )
    """

    # Threshold for using episodes vs raw history
    EPISODE_THRESHOLD_MESSAGES = 20

    def __init__(
        self,
        storage: "AsyncStorage",
        model: str = "auto",
        agent_id: Optional[str] = None,
        consolidator: Optional["MemoryConsolidator"] = None,
        memory_retriever: Optional["MemoryRetriever"] = None,
        llm_service: Optional["LLMService"] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize the context manager.

        Args:
            storage: AsyncStorage instance for all storage operations
            model: Deprecated fallback model name (use llm_service instead)
            agent_id: Agent ID for scoped queries
            consolidator: MemoryConsolidator for episode access
            memory_retriever: MemoryRetriever for emotional memory access
            llm_service: LLMService for model identity and constitutional awareness
            context_builder: Pre-configured ContextBuilder to use (dependency
                injection).  Production callers (KestrelAgent) pass their own
                instance, which has bootstrap files (SOUL.md, AGENTS.md, …)
                loaded into it — that's how the agent's identity flows into
                the system prompt.  When None, a minimal ContextBuilder is
                built here; that fallback is for isolated test contexts and
                will NOT carry bootstrap content, so any production path
                that omits this argument is wrong.
        """
        self.storage = storage
        self._llm_service = llm_service
        self._model_fallback = model
        self._counter = None
        self._counter_model = None
        self.agent_id = agent_id
        self.consolidator = consolidator
        self.memory_retriever = memory_retriever
        # Keep public reference for constitutional awareness (used by build_context)
        self.llm_service = llm_service

        # Initialize sub-components.  Prefer the injected ContextBuilder —
        # that's the one the agent pre-loaded SOUL.md into.  Constructing a
        # fresh one here would leave its BootstrapLoader empty, which is
        # exactly the bug that caused the agent to not know its own name
        # when asked in chat.
        if context_builder is not None:
            self.context_builder = context_builder
        else:
            self.context_builder = ContextBuilder(
                storage=storage,
                consolidator=consolidator,
                llm_service=llm_service,
                model=model,
            )

        # Initialize specialized managers
        self.conversation_manager = ConversationManager(storage, agent_id)
        self.memory_manager = MemoryManager(storage, agent_id, consolidator, memory_retriever)
        self.tool_context_manager = ToolContextManager(
            storage=storage,
            model=self.model,
            agent_id=agent_id,
            llm_service=llm_service,
        )

    @property
    def model(self) -> str:
        """Resolved model ID. Delegates to LLMService if available."""
        if self._llm_service:
            return self._llm_service.get_active_model_id()
        return self._model_fallback

    @model.setter
    def model(self, value: str):
        """Allow direct assignment for backward compatibility."""
        self._model_fallback = value

    @property
    def counter(self) -> TokenCounter:
        """TokenCounter keyed to current model, lazily cached."""
        current_model = self.model
        if self._counter is None or self._counter_model != current_model:
            self._counter = get_token_counter(current_model)
            self._counter_model = current_model
        return self._counter

    @counter.setter
    def counter(self, value):
        """Allow direct assignment for backward compatibility."""
        self._counter = value
        self._counter_model = None

    async def build_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool = True,
        include_memories: bool = True,
        include_rag: bool = True,
        privacy_mode: str = "NORMAL",
        emotional_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None,
        reflection_guidance: Optional[List[str]] = None,
        system_prompt_addendum: Optional[str] = None,
        system_prompt_budget_bytes: Optional[int] = None,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
    ) -> ContextResult:
        """
        Build complete context for an LLM request.

        This is the main entry point for context assembly. It:
        1. Creates an adaptive token budget based on conversation length
        2. Builds the system prompt with constitution
        3. Retrieves and formats conversation history
        4. Adds episode summaries for long conversations
        5. Retrieves emotionally-relevant memories
        6. Searches RAG for relevant documents
        7. Ensures everything fits within token limits

        Args:
            query: The current user query
            constitution: Constitution text to include
            include_briefing: Include session briefing in system prompt
            include_memories: Include emotionally-weighted memories
            include_rag: Include RAG document search
            privacy_mode: Current privacy mode (affects what's retrieved)
            emotional_context: Current emotional state for mood-congruent recall
            conversation_history: Pre-fetched conversation history (e.g., session-filtered).
                                  If None, fetches full history from storage.

        Returns:
            ContextResult with assembled context and metadata
        """
        warnings: List[str] = []

        # Handle EPHEMERAL mode - no retrieval
        if privacy_mode == "EPHEMERAL":
            return await self._build_ephemeral_context(
                query=query,
                constitution=constitution,
                include_briefing=include_briefing,
                system_prompt_addendum=system_prompt_addendum,
                system_prompt_budget_bytes=system_prompt_budget_bytes,
                anchored_doctrine=anchored_doctrine,
            )

        # Use provided history or fetch from storage
        if conversation_history is not None:
            history = conversation_history
        else:
            history = await self.conversation_manager.get_conversation_history()
        message_count = len(history)

        # Get constitutional awareness (state of mind includes prompt adaptation).
        # Resolved BEFORE the budget so the elastic budget can include
        # state-of-mind tokens in the measured mandatory floor.
        prompt_adaptation = None
        state_of_mind = None
        if self.llm_service and hasattr(self.llm_service, 'get_state_of_mind'):
            try:
                state_of_mind = self.llm_service.get_state_of_mind()
                prompt_adaptation = state_of_mind.prompt_adaptation
            except Exception as e:
                logger.warning(f"Failed to get constitutional state of mind: {e}")

        # Measure the non-borrowable mandatory governance floor for the
        # #1309 elastic budget (Emma 2026-05-20). When the floor cannot
        # fit, the elastic budget raises DegradedModeError; surface
        # this as a degraded-mode ContextResult so the caller does not
        # issue the LLM call under a false "normal" status. The guard
        # below is narrow: it only swallows ``TypeError`` from test
        # stubs that mock the token counter (MagicMock returns when
        # casted to int blow up here, not in production). Any other
        # exception — including ``ValueError`` indicating a real
        # measurement error — propagates so a broken measurement path
        # is loud, not silent. Codex round 1 #4.
        raw_mandatory = self.context_builder.measure_mandatory_system_tokens(
            constitution=constitution,
            state_of_mind=state_of_mind,
            prompt_adaptation=prompt_adaptation,
        )
        try:
            mandatory_system_tokens = int(raw_mandatory)
        except TypeError:
            logger.error(
                "measure_mandatory_system_tokens returned non-numeric (type=%s); "
                "treating mandatory floor as 0 — production token counters always "
                "return int, so this signals a test-stub or broken counter wiring",
                type(raw_mandatory).__name__,
            )
            mandatory_system_tokens = 0
        try:
            budget = create_budget(
                self.model,
                message_count,
                elastic=True,
                mandatory_system_tokens=mandatory_system_tokens,
            )
        except DegradedModeError as e:
            logger.error(
                "degraded-mode: %s — returning empty ContextResult; caller "
                "MUST surface this and refuse the LLM call",
                e,
            )
            warnings.append(
                f"DEGRADED MODE: mandatory governance floor ({e.mandatory_system_tokens} "
                f"tokens) does not fit context budget ({e.total_budget} tokens) "
                f"for model {e.model!r}. The LLM call MUST NOT proceed — surface "
                "this to the operator."
            )
            return ContextResult(
                system_prompt="",
                messages=[],
                total_tokens=0,
                budget_summary={"mode": "degraded", "reason": str(e)},
                warnings=warnings,
                degraded_mode=True,
                mandatory_system_tokens=mandatory_system_tokens,
            )

        # 1. Build system prompt. When the caller sets
        # `system_prompt_budget_bytes` (a per-source registration knob
        # threaded through by the SignalDispatcher), route to the
        # priority-aware tracking assembler so the budget actually
        # takes effect. The legacy build_system_prompt is byte-stable
        # for the cache path; the tracking variant intentionally has
        # different bytes (different fence convention) so it's only
        # used when the source explicitly opts in via budget.
        #
        # Codex round-12 P2: the addendum (canary directive) must
        # count toward the budget. Reserve its bytes BEFORE the
        # assembler truncates so the final assembled prompt
        # (assembler output + joiner + addendum) fits within the cap.
        # Codex round-17 P2: route to the tracking assembler when
        # EITHER a budget is set OR anchored doctrine is supplied
        # (the legacy build_system_prompt has no anchored_doctrine
        # parameter). Otherwise full-injection sources without a
        # budget would silently fall back to the legacy path that
        # ignores doctrine.
        injected_clauses_for_audit: Optional[List[str]] = None
        dropped_clauses_for_audit: Optional[List[str]] = None
        if system_prompt_budget_bytes is not None or anchored_doctrine:
            # Effective budget: when caller set a budget, reserve
            # addendum bytes from it (codex round-12); when no
            # budget is set but anchored_doctrine triggered this
            # path (codex round-17), pass None to the assembler so
            # it doesn't truncate.
            effective_budget: Optional[int]
            if system_prompt_budget_bytes is None:
                effective_budget = None
            else:
                reserved = 0
                if system_prompt_addendum:
                    reserved = (
                        len(system_prompt_addendum.encode("utf-8")) + 2
                    )  # 2 bytes for the "\n\n" joiner
                effective_budget = max(
                    1, system_prompt_budget_bytes - reserved
                )
            tracking_result = self.context_builder.build_system_prompt_with_tracking(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=state_of_mind,
                budget_bytes=effective_budget,
                anchored_doctrine=anchored_doctrine,
            )
            system_prompt = tracking_result.prompt
            # Surface tracking back to the caller so the dispatcher
            # can populate signal_log.injected_clauses_json /
            # dropped_clauses_json (codex round-13 P2 fix).
            injected_clauses_for_audit = list(tracking_result.injected_clauses)
            dropped_clauses_for_audit = list(tracking_result.dropped_clauses)
            if system_prompt_addendum:
                system_prompt = f"{system_prompt}\n\n{system_prompt_addendum}"
        else:
            system_prompt = self.context_builder.build_system_prompt(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=state_of_mind,
                system_prompt_addendum=system_prompt_addendum,
            )
        system_tokens = self.counter.count(system_prompt)
        # System is mandatory governance content: we have already
        # committed to sending it. Record usage; if accounting cannot
        # absorb it (system_tokens > local + elastic pool), bump the
        # allocation to match what we're actually sending and log
        # loudly. Better to over-report system usage than to send
        # bytes we don't account for (codex round 1 #1).
        if not budget.use("system", system_tokens):
            allocation = budget.allocations.get("system")
            if allocation is not None:
                allocation.budget = max(allocation.budget, system_tokens)
                allocation.used = system_tokens
            warnings.append(
                f"system content ({system_tokens} tokens) exceeded its slice "
                "plus elastic pool; budget accounting forced to match the "
                "bytes already committed for this turn"
            )
            logger.warning(
                "system slice over budget by %s tokens — forcing allocation "
                "to match what is being sent (no silent drift)",
                system_tokens
                - (budget.allocations.get("system").budget if allocation else 0),
            )

        # Track what we include
        episode_count = 0
        memory_count = 0
        rag_chunks = 0
        # Per-turn retrieved context (memories + RAG). Kept OUT of system_prompt
        # so the system prefix stays stable across turns and prompt caches hit.
        dynamic_blocks: List[str] = []

        # 1b. Inject reflection guidance into system prompt
        if reflection_guidance:
            guidance_text = "\n--- ACTIVE REFLECTION GUIDANCE ---\n"
            guidance_text += "Based on self-reflection, keep these insights in mind:\n"
            for item in reflection_guidance:
                guidance_text += f"- {item}\n"
            guidance_text += "--- END GUIDANCE ---"
            # Codex round-15 P2: when a per-source budget is in
            # effect, the late append of reflection guidance must
            # NOT push the total over the cap. Skip the append if
            # there's no room — the budget contract takes precedence
            # over reflection guidance (operator can raise the cap
            # to admit it back).
            if system_prompt_budget_bytes is not None:
                projected = (
                    len(system_prompt.encode("utf-8"))
                    + 2  # "\n\n" joiner
                    + len(guidance_text.encode("utf-8"))
                )
                if projected > system_prompt_budget_bytes:
                    logger.warning(
                        "Skipping reflection guidance for budgeted "
                        "dispatch (would push prompt %d over cap %d)",
                        projected,
                        system_prompt_budget_bytes,
                    )
                else:
                    guidance_tokens = self.counter.count(guidance_text)
                    budget.use("system", guidance_tokens)
                    system_prompt = f"{system_prompt}\n\n{guidance_text}"
                    logger.info(f"Injected {len(reflection_guidance)} reflection guidance items into prompt")
            else:
                guidance_tokens = self.counter.count(guidance_text)
                budget.use("system", guidance_tokens)
                system_prompt = f"{system_prompt}\n\n{guidance_text}"
                logger.info(f"Injected {len(reflection_guidance)} reflection guidance items into prompt")

        # 1c. Microcompact: clear stale tool results (zero-cost, no LLM)
        microcompact_savings = self._microcompact_tool_results(history)
        if microcompact_savings > 0:
            logger.info(f"Microcompact cleared {microcompact_savings} stale tool results")

        # Finalize the system slice: any unused budget above the
        # mandatory floor flows into the elastic pool so later sections
        # (episodes, memories, RAG, history) can borrow it. The
        # mandatory floor is preserved — never returned to the pool.
        if hasattr(budget, "mark_section_finalized"):
            budget.mark_section_finalized("system")

        # 2. Add episodes for long conversations.
        # Use the get/format split (#1308) so episode_count is an
        # accurate ``len(episodes)`` instead of the legacy
        # ``"**".count() // 2`` heuristic — the formatted block contains
        # bold markers for emotional-arc lines and other ``**``-bearing
        # substrings that made the heuristic over- and under-count
        # depending on episode content.
        if message_count >= self.EPISODE_THRESHOLD_MESSAGES and self.consolidator:
            episodes = await self.context_builder.get_episodes_for_context(
                max_tokens=budget.episodes,
                max_episodes=5,
            )
            episode_context = self.context_builder.format_episodes_for_context(episodes)
            if episode_context:
                # Codex round-15 P2: same budget guard as reflection
                # guidance. Skip episode append if it would push the
                # final prompt over the per-source cap.
                if system_prompt_budget_bytes is not None:
                    projected = (
                        len(system_prompt.encode("utf-8"))
                        + 2
                        + len(episode_context.encode("utf-8"))
                    )
                    if projected > system_prompt_budget_bytes:
                        logger.warning(
                            "Skipping episode context for budgeted "
                            "dispatch (would push prompt %d over cap %d)",
                            projected,
                            system_prompt_budget_bytes,
                        )
                        episode_context = None

                if episode_context:
                    episode_tokens = self.counter.count(episode_context)
                    # Only append episodes when the budget can absorb
                    # them — otherwise the block is dropped to preserve
                    # the accounting invariant (codex round 1 #1).
                    # The selector inside ``get_episodes_for_context``
                    # already capped by ``budget.episodes``; this is a
                    # defensive check against pool exhaustion.
                    if budget.use("episodes", episode_tokens, items=len(episodes)):
                        system_prompt = f"{system_prompt}\n\n{episode_context}"
                        episode_count = len(episodes)
                        logger.debug(f"Added {episode_count} episodes to context")
                    else:
                        warnings.append(
                            f"episode block ({episode_tokens} tokens) skipped — "
                            "exceeded episodes slice plus elastic pool"
                        )
                        logger.warning(
                            "episode block skipped: %s tokens did not fit episodes "
                            "slice plus pool",
                            episode_tokens,
                        )
        # Episodes finalized — release any unused episode budget into
        # the elastic pool for later sections.
        if hasattr(budget, "mark_section_finalized"):
            budget.mark_section_finalized("episodes")

        # 3. Retrieve emotionally-weighted memories (placed in dynamic user
        # context, not system, so the system prefix stays cacheable).
        if include_memories and self.memory_retriever:
            try:
                memories = await self.memory_manager.retrieve_memories(
                    query=query,
                    max_tokens=budget.memories,
                    counter=self.counter,
                    emotional_context=emotional_context
                )
                if memories:
                    memory_tokens = self.counter.count(memories)
                    if budget.can_fit("memories", memory_tokens):
                        budget.use("memories", memory_tokens)
                        dynamic_blocks.append(f"<memories>\n{memories}\n</memories>")
                        memory_count = memories.count("[Memory")
                        logger.debug(f"Added {memory_count} memories to dynamic context")
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                warnings.append(f"Memory retrieval unavailable: {e}")
        # Memories finalized — release slack into the elastic pool.
        if hasattr(budget, "mark_section_finalized"):
            budget.mark_section_finalized("memories")

        # 4. Retrieve RAG context (placed in dynamic user context, not system).
        if include_rag:
            try:
                rag_context = await self.context_builder.retrieve_context(query)
                if rag_context:
                    rag_tokens = self.counter.count(rag_context)
                    if budget.can_fit("rag", rag_tokens):
                        budget.use("rag", rag_tokens)
                        dynamic_blocks.append(
                            "<documents>\n"
                            f"{rag_context}\n"
                            "</documents>"
                        )
                        rag_chunks = rag_context.count("[Document") or rag_context.count("Source:")
                        logger.debug(f"Added {rag_chunks} RAG chunks to dynamic context")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                warnings.append(f"Document search unavailable: {e}")
        # RAG finalized — release slack into the pool so the history
        # slice (the section the silent-prune correctness hole hurts
        # most until C ships) can borrow it.
        if hasattr(budget, "mark_section_finalized"):
            budget.mark_section_finalized("rag")

        # Assemble the per-turn retrieved-context block. Empty string when
        # nothing was retrieved — caller can use this as-is in a format()
        # without producing dangling wrapper tags.
        if dynamic_blocks:
            dynamic_user_context = (
                "<retrieved_context>\n"
                + "\n".join(dynamic_blocks)
                + "\n</retrieved_context>"
            )
        else:
            dynamic_user_context = ""

        # 5. Format conversation history with remaining budget. When the
        # elastic budget (#1309) is in use, history sizes against its
        # *effective* ceiling (own remaining + pool) so released slack
        # from finalized sections actually grows the conversation slice
        # — codex round 1 #2 caught the previous version capping at the
        # static ``budget.history`` and never asking for the slack.
        if hasattr(budget, "effective_budget"):
            history_max_tokens = budget.effective_budget("history")
        else:
            history_max_tokens = budget.history
        formatted_history = self.context_builder.format_conversation_history(
            history=history,
            max_tokens=history_max_tokens,
        )
        history_tokens = self.counter.count_messages(formatted_history)
        if not budget.use("history", history_tokens, items=len(formatted_history)):
            # Pool exhausted mid-history; trim the oldest until it fits.
            # The legacy post-budget prune below still runs as final
            # safety, but try a soft pre-prune here so warnings are
            # accurate (codex round 1 #1).
            logger.warning(
                "history use() rejected %s tokens — relying on legacy "
                "post-budget auto-prune to bring back under budget",
                history_tokens,
            )

        # Check if we had to truncate significantly
        if len(formatted_history) < len(history) * 0.5:
            warnings.append(
                f"History truncated: {len(formatted_history)}/{len(history)} messages"
            )

        # Pre-send budget enforcement: if total exceeds budget, drop oldest history
        if budget.total_used > budget.total_budget and len(formatted_history) > 1:
            overage = budget.total_used - budget.total_budget
            logger.warning(
                f"Context budget exceeded by {overage} tokens — auto-pruning oldest history"
            )
            pruned_tokens = 0
            while formatted_history and pruned_tokens < overage:
                dropped = formatted_history.pop(0)  # Drop oldest
                dropped_tokens = self.counter.count(dropped.get("content", "")) + 4
                pruned_tokens += dropped_tokens
            # Update budget tracking
            new_history_tokens = self.counter.count_messages(formatted_history)
            alloc = budget.allocations["history"]
            alloc.used = new_history_tokens
            alloc.items = len(formatted_history)
            warnings.append(
                f"Auto-pruned {pruned_tokens} tokens from history to fit budget"
            )

        logger.info(
            f"Context built: {budget.total_used}/{budget.total_budget} tokens "
            f"({len(formatted_history)} msgs, {episode_count} episodes, "
            f"{memory_count} memories, {rag_chunks} docs)"
        )

        # Codex round-14 P2: publish per-task tracking via ContextVar
        # so concurrent COGNITION dispatches don't race on a shared
        # agent attribute. Dispatcher reads via
        # `get_current_injection_tracking()` after process_input.
        _INJECTION_TRACKING_VAR.set(
            (injected_clauses_for_audit, dropped_clauses_for_audit)
        )

        return ContextResult(
            system_prompt=system_prompt,
            messages=formatted_history,
            total_tokens=budget.total_used,
            budget_summary=budget.get_summary(),
            episode_count=episode_count,
            memory_count=memory_count,
            rag_chunks=rag_chunks,
            warnings=warnings,
            dynamic_user_context=dynamic_user_context,
            injected_clauses=injected_clauses_for_audit,
            dropped_clauses=dropped_clauses_for_audit,
            degraded_mode=False,
            mandatory_system_tokens=mandatory_system_tokens,
        )

    async def _build_ephemeral_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool,
        system_prompt_addendum: Optional[str] = None,
        system_prompt_budget_bytes: Optional[int] = None,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
    ) -> ContextResult:
        """
        Build minimal context for EPHEMERAL privacy mode.

        In EPHEMERAL mode, no history or memories are retrieved.
        Only the system prompt and constitution are included.
        """
        # Get constitutional awareness (state of mind includes prompt adaptation)
        prompt_adaptation = None
        state_of_mind = None
        if self.llm_service and hasattr(self.llm_service, 'get_state_of_mind'):
            try:
                state_of_mind = self.llm_service.get_state_of_mind()
                prompt_adaptation = state_of_mind.prompt_adaptation
            except Exception as e:
                logger.warning(f"Failed to get constitutional state of mind: {e}")

        # The EPHEMERAL MODE notice is fixed text appended after the
        # budget-aware assembly. Codex round-13 P2 caught that we
        # must reserve its bytes too, otherwise the notice can push
        # the final prompt over the configured budget.
        ephemeral_notice = (
            "--- EPHEMERAL MODE ACTIVE ---\n"
            "This conversation is not being recorded. "
            "No history or memories are available.\n"
            "--- END NOTICE ---"
        )

        ephemeral_tracking = None
        injected_clauses_for_audit: Optional[List[str]] = None
        dropped_clauses_for_audit: Optional[List[str]] = None
        if system_prompt_budget_bytes is not None or anchored_doctrine:
            # Reserve addendum + ephemeral notice + their joiners.
            effective_budget: Optional[int]
            if system_prompt_budget_bytes is None:
                effective_budget = None
            else:
                reserved = 0
                if system_prompt_addendum:
                    reserved += (
                        len(system_prompt_addendum.encode("utf-8")) + 2
                    )
                reserved += len(ephemeral_notice.encode("utf-8")) + 2
                effective_budget = max(
                    1, system_prompt_budget_bytes - reserved
                )
            ephemeral_tracking = self.context_builder.build_system_prompt_with_tracking(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=state_of_mind,
                budget_bytes=effective_budget,
                anchored_doctrine=anchored_doctrine,
            )
            system_prompt = ephemeral_tracking.prompt
            injected_clauses_for_audit = list(ephemeral_tracking.injected_clauses)
            dropped_clauses_for_audit = list(ephemeral_tracking.dropped_clauses)
            if system_prompt_addendum:
                system_prompt = (
                    f"{system_prompt}\n\n{system_prompt_addendum}"
                )
        else:
            system_prompt = self.context_builder.build_system_prompt(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=state_of_mind,
                system_prompt_addendum=system_prompt_addendum,
            )

        # Append the ephemeral notice (already accounted for in the
        # reserved budget above when budget_bytes was set).
        system_prompt = (
            f"{system_prompt}\n\n"
            "--- EPHEMERAL MODE ACTIVE ---\n"
            "This conversation is not being recorded. "
            "No history or memories are available.\n"
            "--- END NOTICE ---"
        )

        tokens = self.counter.count(system_prompt)

        # Same per-task tracking publish as the non-ephemeral path.
        _INJECTION_TRACKING_VAR.set(
            (injected_clauses_for_audit, dropped_clauses_for_audit)
        )

        return ContextResult(
            system_prompt=system_prompt,
            messages=[],
            total_tokens=tokens,
            budget_summary={"mode": "ephemeral"},
            warnings=["EPHEMERAL mode: no history available"],
            injected_clauses=injected_clauses_for_audit,
            dropped_clauses=dropped_clauses_for_audit,
        )

    # Delegate to ConversationManager
    async def compress_session(self, llm_service, preserve_recent: int = 10, force: bool = False) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.compress_session(
            llm_service, self.counter, preserve_recent, force
        )

    async def check_compression_needed(self, utilization_threshold: float = 70.0) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.check_compression_needed(
            self.counter, self.model, utilization_threshold
        )

    async def get_messages_for_selection(self, mode: str, criteria: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.get_messages_for_selection(mode, criteria, limit)

    async def mark_messages(self, message_ids: List[int], action: str, reason: str = "") -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.mark_messages(message_ids, action, reason)

    async def exclude_messages(self, message_ids: List[int], reason: str) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.exclude_messages(message_ids, reason)

    async def restore_messages(self, message_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.restore_messages(message_ids)

    async def summarize_messages(self, llm_service, message_ids: List[int], preserve_key_facts: bool = True) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.summarize_messages(
            llm_service, self.counter, message_ids, preserve_key_facts
        )

    # Delegate to MemoryManager
    async def check_episode_needed(self, session_messages: int = 0) -> bool:
        """Delegate to MemoryManager."""
        return await self.memory_manager.check_episode_needed(session_messages)

    async def create_episode_if_needed(self, session_messages: int = 0, force: bool = False) -> Optional[Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.create_episode_if_needed(session_messages, force)

    async def stash_messages(self, message_ids: Optional[List[int]] = None, name: Optional[str] = None, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_messages(message_ids, name, last_n)

    async def stash_pop(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_pop(stash_id)

    async def stash_apply(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_apply(stash_id)

    async def stash_list(self) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_list()

    async def stash_drop(self, stash_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_drop(stash_id)

    async def stash_save(self, stash_id: Optional[str] = None, name: Optional[str] = None, summary: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_save(stash_id, name, summary, tags)

    async def stash_peek(self, stash_id: Optional[str] = None, max_chars: int = 5000) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.stash_peek(stash_id, max_chars)

    async def hierarchical_compress(self, llm_service, chunk_size: int = 4000, preserve_recent: int = 5, max_depth: int = 3) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.hierarchical_compress(
            llm_service, self.counter, chunk_size, preserve_recent, max_depth
        )

    # Delegate to ToolContextManager
    async def get_status(self, history: Optional[list] = None, context_stats=None) -> Dict[str, Any]:
        """Delegate to ToolContextManager. Pass session-filtered history for accurate reporting."""
        return await self.tool_context_manager.get_status(self.counter, history=history, context_stats=context_stats)

    def get_budget_status(self, message_count: int = 0) -> Dict[str, Any]:
        """Delegate to ToolContextManager."""
        return self.tool_context_manager.get_budget_status(message_count)

    # Helper methods exposed for testing
    def _build_message_chunks(self, messages: List[Dict], chunk_size: int) -> List[str]:
        """Delegate to MemoryManager."""
        return self.memory_manager._build_message_chunks(messages, chunk_size)

    # --- Microcompact: zero-cost tool result clearing ---

    MICROCOMPACT_KEEP_RECENT = int(os.environ.get("KESTREL_MICROCOMPACT_KEEP_RECENT", "5"))

    def _microcompact_tool_results(self, history: List[Dict]) -> int:
        """
        Clear stale tool result content from conversation history.

        Replaces old tool result content with JSON markers while preserving
        tool_call_id pairing (required by LLM APIs). The most recent N
        tool results are kept intact.

        This runs BEFORE format_conversation_history() normalizes roles,
        so role="tool" is still identifiable.

        Args:
            history: Conversation history (mutated in place)

        Returns:
            Number of tool results cleared
        """
        keep_recent = self.MICROCOMPACT_KEEP_RECENT
        if keep_recent < 1:
            keep_recent = 1

        # Collect indices of tool result messages (preserving order)
        tool_indices = []
        for i, msg in enumerate(history):
            if msg.get("role") != "tool":
                continue
            # Skip protected or already excluded messages
            meta = msg.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    # Can't read protection flags — assume protected, skip
                    continue
            if meta.get("context_priority") == "protected":
                continue
            if meta.get("excluded_from_context"):
                continue
            if meta.get("decay_protected"):
                continue
            tool_indices.append(i)

        if len(tool_indices) <= keep_recent:
            return 0

        # Keep the last N, clear the rest
        to_clear = tool_indices[:-keep_recent]
        cleared = 0
        now = datetime.now(timezone.utc).isoformat()

        for idx in to_clear:
            msg = history[idx]
            content = msg.get("content", "")

            # Already cleared?
            if isinstance(content, str) and content.startswith('{"cleared":'):
                continue

            # Build informative marker
            tool_name = ""
            summary = ""
            meta = msg.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            tool_name = meta.get("tool_name", "")

            # Extract summary from content (first 100 chars of the result)
            if isinstance(content, str):
                summary = content[:100].replace('"', '\\"')

            marker = json.dumps({
                "cleared": True,
                "tool_name": tool_name,
                "summary": summary,
                "cleared_at": now,
            })
            msg["content"] = marker
            cleared += 1

        return cleared
