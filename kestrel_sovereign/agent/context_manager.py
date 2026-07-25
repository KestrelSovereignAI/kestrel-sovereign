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

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

from .context_builder import ContextBuilder
from .context_stages import (
    ContextAssembly,
    SectionDestination,
    SectionResult,
    build_episode_section,
    build_memory_section,
    build_rag_section,
    build_reflection_guidance_block,
    compute_lumpy_anchor,
    compute_pruned_span,
    emit_content_for_msg,
    finalize_section,
    microcompact_tool_results,
    EPHEMERAL_NOTICE,
)
from .token_counter import TokenCounter, get_token_counter
from .token_budget import TokenBudget, create_budget, DegradedModeError
from .conversation_manager import ConversationManager
from .salvage import (
    SalvageReason,
    SalvageWriteError,
    get_pending_count,
    is_durable_salvage_enabled,
    salvage_messages,
)
from .memory_manager import MemoryManager
from .tool_context_manager import ToolContextManager

if TYPE_CHECKING:
    from collections import OrderedDict

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


# Relevance-gate defaults (#1404). Conservative floors — set high
# enough to drop weak matches but low enough that genuinely relevant
# content still surfaces. Override via ``[retrieval]`` in kestrel.toml.
_RETRIEVAL_DEFAULTS = {
    "memory_min_score": 0.3,
    "memory_min_relevance": 0.2,
    "rag_min_score": 0.5,
}
_RETRIEVAL_CONFIG_CACHE: Optional[Dict[str, float]] = None


def _retrieval_config() -> Dict[str, float]:
    """Resolve the relevance-gate config from kestrel.toml (#1404).

    Reads the ``[retrieval]`` block on first call and caches the merged
    config for subsequent turns. Missing keys fall back to
    :data:`_RETRIEVAL_DEFAULTS`. Missing config file or unreadable TOML
    falls back silently to defaults — config is optional, not required.

    Cache busts on test isolation via ``reset_retrieval_config_cache``.
    """
    global _RETRIEVAL_CONFIG_CACHE
    if _RETRIEVAL_CONFIG_CACHE is not None:
        return _RETRIEVAL_CONFIG_CACHE

    merged = dict(_RETRIEVAL_DEFAULTS)
    try:
        from pathlib import Path
        try:
            import tomllib  # Python 3.11+
        except ImportError:  # pragma: no cover — pre-3.11 path
            import tomli as tomllib  # type: ignore

        try:
            from kestrel_sovereign.paths import project_dir
            root = Path(project_dir())
        except Exception:
            root = Path.cwd()
        toml_path = root / "kestrel.toml"
        if toml_path.is_file():
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
            section = data.get("retrieval", {}) or {}
            for key in _RETRIEVAL_DEFAULTS:
                if key in section and isinstance(section[key], (int, float)):
                    merged[key] = float(section[key])
    except Exception as e:
        logger.debug(f"Falling back to retrieval defaults: {e}")

    _RETRIEVAL_CONFIG_CACHE = merged
    return merged


def reset_retrieval_config_cache() -> None:
    """Test hook: drop the cached relevance-gate config so the next
    call re-reads ``kestrel.toml`` (#1404)."""
    global _RETRIEVAL_CONFIG_CACHE
    _RETRIEVAL_CONFIG_CACHE = None


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
    # Exact per-turn snapshot used to derive prompt adaptation. The turn
    # orchestrator reuses this same object for governance-delta notices so the
    # two views cannot drift when model state changes during context assembly.
    state_of_mind: Any = None


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

        # C / #1311 durable salvage worker. Lazily started — the
        # janitor only runs when ``start_salvage_worker()`` is
        # invoked by the agent's lifecycle (KestrelAgent.startup
        # hook). When ``is_durable_salvage_enabled()`` is False the
        # worker stays None and the build_context salvage branch
        # is a no-op. Codex round 1 #3 caught the earlier version
        # that left this slot empty in production.
        self.salvage_worker = None
        self._salvage_worker_started = False

    async def start_salvage_worker(self) -> None:
        """Start the C / #1311 durable-salvage background worker.

        Idempotent. Called by the agent's startup hook. When the
        feature flag is disabled, this is a no-op so legacy
        deployments do not spin a janitor task they will never use.
        """
        from .salvage import SalvageWorker, is_durable_salvage_enabled

        if self._salvage_worker_started:
            return
        if not is_durable_salvage_enabled():
            return
        conv_store = getattr(
            self.conversation_manager, "_get_conversation_store", lambda: None
        )()
        if conv_store is None:
            logger.warning(
                "salvage worker not started: no conversation store available"
            )
            return
        if self.llm_service is None or not hasattr(self.llm_service, "generate"):
            logger.warning(
                "salvage worker not started: llm_service.generate not available"
            )
            return

        async def _llm_completion(**kwargs):
            # Adapter to LLMService.generate's canonical signature
            # (user_prompt + system_prompt). The SalvageWorker
            # introspects the callable and prefers user_prompt,
            # so we accept either.
            user_prompt = kwargs.pop("user_prompt", None) or kwargs.pop("prompt", "")
            system_prompt = kwargs.pop("system_prompt", None)
            return await self.llm_service.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

        self.salvage_worker = SalvageWorker(
            conv_store=conv_store,
            llm_completion=_llm_completion,
        )
        await self.salvage_worker.start()
        self._salvage_worker_started = True
        logger.info("salvage worker started (C / #1311 feature flag enabled)")

    async def stop_salvage_worker(self) -> None:
        """Stop the salvage worker; called by the agent's shutdown hook."""
        if self.salvage_worker is not None:
            await self.salvage_worker.stop()
        self.salvage_worker = None
        self._salvage_worker_started = False

    @property
    def model(self) -> str:
        """Resolved model ID, route-qualified when a route is active.

        Returns the canonical ``"<vendor>:<route>/<model>"`` form when
        available (e.g. ``"openai:plan/gpt-5.5"``), falling back to the
        bare model id when no route is configured. The route-qualified
        form is what the TokenBudget and TokenCounter need to look up
        the *per-turn* context cap — which can be much lower than the
        model's full context window (ChatGPT-subscription Plus is the
        canonical case; #1395).

        ``TokenCounter.get_context_limit()`` already normalizes by
        trying the route-qualified key first and then the bare model
        id, so callers that registered context limits under the bare
        model id continue to work.
        """
        if self._llm_service:
            if hasattr(self._llm_service, "get_active_model_selection"):
                try:
                    selection = self._llm_service.get_active_model_selection()
                    qualified = selection.get("model") if selection else None
                    if qualified:
                        return qualified
                except Exception as e:
                    logger.debug(
                        "get_active_model_selection failed (%s); "
                        "falling back to get_active_model_id",
                        e,
                    )
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
        # One typed state object per call — never a shared instance — so
        # concurrent COGNITION dispatches cannot cross-contaminate counters
        # or results. Per-task injection tracking is published via the
        # ContextVar only on the successful path at the end of this method.
        assembly = ContextAssembly()

        # Resolve constitutional awareness once. Prompt adaptation is stable
        # top-level context; mutable StateOfMind fields travel as append-only
        # operator facts. Returning this exact object in ContextResult keeps
        # those two views on the same per-turn snapshot.
        state_of_mind, prompt_adaptation = (
            self._resolve_state_of_mind_snapshot()
        )

        # Handle EPHEMERAL mode - no retrieval
        if privacy_mode == "EPHEMERAL":
            return await self._build_ephemeral_context(
                query=query,
                constitution=constitution,
                include_briefing=include_briefing,
                system_prompt_addendum=system_prompt_addendum,
                system_prompt_budget_bytes=system_prompt_budget_bytes,
                anchored_doctrine=anchored_doctrine,
                state_of_mind=state_of_mind,
                prompt_adaptation=prompt_adaptation,
            )

        # Use provided history or fetch from storage
        if conversation_history is not None:
            history = conversation_history
        else:
            history = await self.conversation_manager.get_conversation_history()
        message_count = len(history)

        # Measure the non-borrowable mandatory governance floor and build the
        # #1309 elastic budget (Emma 2026-05-20). A floor that cannot fit
        # raises DegradedModeError → surface a degraded-mode ContextResult so
        # the caller does not issue the LLM call under a false "normal" status.
        mandatory_system_tokens = self._measure_mandatory_system_tokens(
            constitution, prompt_adaptation
        )
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
            assembly.warnings.append(
                f"DEGRADED MODE: mandatory governance floor ({e.mandatory_system_tokens} "
                f"tokens) does not fit context budget ({e.total_budget} tokens) "
                f"for model {e.model!r}. The LLM call MUST NOT proceed — surface "
                "this to the operator."
            )
            return self._degraded_result(
                assembly,
                reason=str(e),
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
            )

        # 1. Assemble the stable system prefix (constitution/identity/doctrine)
        # and record its usage. Kept separate from the per-turn dynamic user
        # context by construction (ContextAssembly). The tracking assembler is
        # used when a per-source byte budget is set OR anchored doctrine is
        # supplied; otherwise the byte-stable legacy assembler.
        (
            assembly.system_prompt,
            assembly.injected_clauses,
            assembly.dropped_clauses,
        ) = self._assemble_system_prompt(
            constitution=constitution,
            include_briefing=include_briefing,
            prompt_adaptation=prompt_adaptation,
            system_prompt_addendum=system_prompt_addendum,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
            anchored_doctrine=anchored_doctrine,
        )
        self._record_system_usage(assembly, budget)

        # 1b. Reflection guidance (into the system prompt, budget-gated).
        self._apply_reflection_guidance(
            assembly,
            budget,
            reflection_guidance,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
        )

        # 1c. Microcompact: clear stale tool results (zero-cost, no LLM).
        microcompact_savings = self._microcompact_tool_results(history)
        if microcompact_savings > 0:
            logger.info(
                f"Microcompact cleared {microcompact_savings} stale tool results"
            )

        # Finalize the system slice: any unused budget above the mandatory
        # floor flows into the elastic pool so later sections (episodes,
        # memories, RAG, history) can borrow it. The mandatory floor is
        # preserved — never returned to the pool.
        finalize_section(budget, "system")

        # 2. Episodes for long conversations (into the system prompt). The
        # get/format split (#1308) yields an accurate ``len(episodes)`` count.
        episode_result = await self._produce_episodes(
            assembly,
            budget,
            message_count,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
        )
        self._commit_episodes(assembly, budget, episode_result)
        finalize_section(budget, "episodes")

        # Active TodoFeature rollups are injected via the always-on operational
        # pre-turn block — ``preturn_state._active_todo_section`` (#1907) — so
        # they survive signal wakes/restarts even when the optional proactive
        # [preturn_state] block is off, rather than only here in the
        # query-dependent context path.

        # Relevance gate (#1404): trivial turns (greetings, sign-offs,
        # bang/slash commands, very-short utterances) skip memory + RAG
        # retrieval entirely. The cost is per-call retrieval cycles and,
        # more importantly, an empty ``dynamic_user_context`` — so the
        # rendered transport form for "hi" carries no ``<retrieved_context>``
        # block and the next turn's retrieval doesn't see stamped noise.
        from kestrel_sovereign.agent.turn_classifier import is_trivial_turn
        retrieval_cfg = _retrieval_config()
        trivial_turn = is_trivial_turn(query)
        if trivial_turn:
            logger.debug(
                "Trivial turn classified by turn_classifier — "
                "skipping memory + RAG retrieval (#1404)"
            )

        # 3. Emotionally-weighted memories (into dynamic user context, not
        # system, so the system prefix stays cacheable). Access rehearsal is
        # recorded only after the block is actually inserted.
        memory_result = await self._produce_memories(
            budget,
            query,
            emotional_context,
            retrieval_cfg,
            include_memories=include_memories,
            trivial_turn=trivial_turn,
        )
        self._commit_dynamic_section(assembly, budget, memory_result)
        # Access rehearsal is recorded only after the block is actually
        # inserted, and only for structured memory blocks (matching the
        # legacy ``isinstance(memories, RetrievedMemoryBlock)`` gate).
        if (
            memory_result is not None
            and memory_result.committed
            and memory_result.is_memory_block
        ):
            # Access rehearsal is non-critical bookkeeping: the memory block
            # is already committed to the dynamic context above, so a failure
            # here must not fail the whole build. The legacy broad
            # memory-retrieval handler swallowed this (the ``record_accesses``
            # await lived inside its ``try/except``); the #2523 decomposition
            # moved the await out here, dropping that guard. Restore
            # warn-and-continue around the bookkeeping only — retrieval, budget,
            # and salvage failure contracts are untouched.
            try:
                await self.memory_retriever.record_accesses(
                    memory_result.message_ids, self.agent_id
                )
            except Exception as e:
                logger.warning(f"Memory access bookkeeping failed: {e}")
                assembly.warnings.append(
                    f"Memory access bookkeeping unavailable: {e}"
                )
        finalize_section(budget, "memories")

        # 4. RAG documents (into dynamic user context, not system).
        rag_result = await self._produce_rag(
            budget,
            query,
            retrieval_cfg,
            include_rag=include_rag,
            trivial_turn=trivial_turn,
        )
        self._commit_dynamic_section(assembly, budget, rag_result)
        finalize_section(budget, "rag")

        # 5. Format conversation history with the remaining (elastic) budget,
        # lumpy-anchored for a cache-stable prefix, then reconciled to fit.
        self._apply_history(assembly, budget, history)

        # === C / #1311: durable salvage of pruned spans ===
        # No model-visible prune may return before its durable salvage
        # evidence commits. This is the single fail-closed finalization
        # boundary: a salvage write failure — or an unreachable store while
        # the feature flag is on — drops into degraded mode rather than
        # silently letting bytes leave the model view (Emma 2026-05-20).
        degraded = await self._finalize_salvage(
            assembly,
            history,
            mandatory_system_tokens=mandatory_system_tokens,
            state_of_mind=state_of_mind,
        )
        if degraded is not None:
            return degraded

        logger.info(
            f"Context built: {budget.total_used}/{budget.total_budget} tokens "
            f"({len(assembly.formatted_history)} msgs, {assembly.episode_count} episodes, "
            f"{assembly.memory_count} memories, {assembly.rag_chunks} docs)"
        )

        # Codex round-14 P2: publish per-task tracking via ContextVar
        # so concurrent COGNITION dispatches don't race on a shared
        # agent attribute. Dispatcher reads via
        # `get_current_injection_tracking()` after process_input.
        _INJECTION_TRACKING_VAR.set(
            (assembly.injected_clauses, assembly.dropped_clauses)
        )

        return ContextResult(
            system_prompt=assembly.system_prompt,
            messages=assembly.formatted_history,
            total_tokens=budget.total_used,
            budget_summary=budget.get_summary(),
            episode_count=assembly.episode_count,
            memory_count=assembly.memory_count,
            rag_chunks=assembly.rag_chunks,
            warnings=assembly.warnings,
            dynamic_user_context=assembly.dynamic_user_context,
            injected_clauses=assembly.injected_clauses,
            dropped_clauses=assembly.dropped_clauses,
            degraded_mode=False,
            mandatory_system_tokens=mandatory_system_tokens,
            state_of_mind=state_of_mind,
        )

    # ------------------------------------------------------------------
    # build_context stages — each produces/commits one section. Content
    # vocabulary is shared with the measurement path via ``context_stages``;
    # the elastic finalization boundary is applied by the orchestrator, not
    # by these stages (they never call ``mark_section_finalized``).
    # ------------------------------------------------------------------

    def _measure_mandatory_system_tokens(
        self, constitution: str, prompt_adaptation: Any
    ) -> int:
        """Measure the non-borrowable mandatory governance floor (#1309).

        The narrow ``TypeError`` guard only swallows test stubs that mock
        the token counter (a MagicMock casted to int blows up here, not in
        production). Any other exception — including a real ``ValueError``
        measurement error — propagates so a broken path is loud, not silent
        (codex round 1 #4).
        """
        raw_mandatory = self.context_builder.measure_mandatory_system_tokens(
            constitution=constitution,
            state_of_mind=None,
            prompt_adaptation=prompt_adaptation,
        )
        try:
            return int(raw_mandatory)
        except TypeError:
            logger.error(
                "measure_mandatory_system_tokens returned non-numeric (type=%s); "
                "treating mandatory floor as 0 — production token counters always "
                "return int, so this signals a test-stub or broken counter wiring",
                type(raw_mandatory).__name__,
            )
            return 0

    def _assemble_system_prompt(
        self,
        *,
        constitution: str,
        include_briefing: bool,
        prompt_adaptation: Any,
        system_prompt_addendum: Optional[str],
        system_prompt_budget_bytes: Optional[int],
        anchored_doctrine: Optional["OrderedDict[str, str]"],
    ) -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
        """Assemble the stable system prefix and optional injection tracking.

        Routes to the priority-aware tracking assembler when the caller sets
        a per-source byte budget OR supplies anchored doctrine (the legacy
        ``build_system_prompt`` has no ``anchored_doctrine`` parameter);
        otherwise uses the byte-stable legacy assembler so the cache prefix
        stays identical for legacy callers. When budgeting, the addendum's
        bytes are reserved BEFORE the assembler truncates (codex round-12 P2)
        so the final ``assembler output + joiner + addendum`` fits the cap.

        Returns ``(system_prompt, injected_clauses, dropped_clauses)``; the
        clause lists are ``None`` for the legacy path.
        """
        injected_clauses: Optional[List[str]] = None
        dropped_clauses: Optional[List[str]] = None
        if system_prompt_budget_bytes is not None or anchored_doctrine:
            effective_budget: Optional[int]
            if system_prompt_budget_bytes is None:
                # anchored_doctrine triggered this path with no budget
                # (codex round-17 P2): pass None so nothing truncates.
                effective_budget = None
            else:
                reserved = 0
                if system_prompt_addendum:
                    reserved = (
                        len(system_prompt_addendum.encode("utf-8")) + 2
                    )  # 2 bytes for the "\n\n" joiner
                effective_budget = max(1, system_prompt_budget_bytes - reserved)
            tracking_result = self.context_builder.build_system_prompt_with_tracking(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=None,
                budget_bytes=effective_budget,
                anchored_doctrine=anchored_doctrine,
            )
            system_prompt = tracking_result.prompt
            injected_clauses = list(tracking_result.injected_clauses)
            dropped_clauses = list(tracking_result.dropped_clauses)
            if system_prompt_addendum:
                system_prompt = f"{system_prompt}\n\n{system_prompt_addendum}"
        else:
            system_prompt = self.context_builder.build_system_prompt(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=None,
                system_prompt_addendum=system_prompt_addendum,
            )
        return system_prompt, injected_clauses, dropped_clauses

    def _record_system_usage(
        self, assembly: ContextAssembly, budget: TokenBudget
    ) -> None:
        """Record the (mandatory) system content against the budget.

        We have already committed to sending the system prompt; if accounting
        cannot absorb it, bump the allocation to match the bytes being sent
        and warn loudly rather than let usage drift silently (codex round 1
        #1).
        """
        system_tokens = self.counter.count(assembly.system_prompt)
        if not budget.use("system", system_tokens):
            allocation = budget.allocations.get("system")
            if allocation is not None:
                allocation.budget = max(allocation.budget, system_tokens)
                allocation.used = system_tokens
            assembly.warnings.append(
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

    def _apply_reflection_guidance(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        reflection_guidance: Optional[List[str]],
        *,
        system_prompt_budget_bytes: Optional[int],
    ) -> None:
        """Append reflection guidance to the system prompt (budget-gated).

        When a per-source byte budget is in effect the late append must not
        push the total over the cap — the budget contract takes precedence
        over reflection guidance (codex round-15 P2).
        """
        if not reflection_guidance:
            return
        guidance_text = build_reflection_guidance_block(reflection_guidance)
        if system_prompt_budget_bytes is not None:
            projected = (
                len(assembly.system_prompt.encode("utf-8"))
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
                return
        guidance_tokens = self.counter.count(guidance_text)
        budget.use("system", guidance_tokens)
        assembly.system_prompt = f"{assembly.system_prompt}\n\n{guidance_text}"
        logger.info(
            f"Injected {len(reflection_guidance)} reflection guidance items into prompt"
        )

    async def _produce_episodes(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        message_count: int,
        *,
        system_prompt_budget_bytes: Optional[int],
    ) -> Optional[SectionResult]:
        """Retrieve + format episode summaries for long conversations.

        Applies the per-source byte-budget projection guard before proposing
        the append (codex round-15 P2); budget acceptance is decided by the
        commit step. Returns ``None`` when episodes don't apply or are guarded
        out.
        """
        if not (
            message_count >= self.EPISODE_THRESHOLD_MESSAGES and self.consolidator
        ):
            return None
        episodes = await self.context_builder.get_episodes_for_context(
            max_tokens=budget.episodes,
            max_episodes=5,
        )
        episode_context = self.context_builder.format_episodes_for_context(episodes)
        if not episode_context:
            return None
        if system_prompt_budget_bytes is not None:
            projected = (
                len(assembly.system_prompt.encode("utf-8"))
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
                return None
        # Shared with ``measure_context_breakdown`` via ``context_stages``
        # so the two assemblers cannot disagree on the episode block's
        # token cost or ``len(episodes)`` count.
        return build_episode_section(
            episode_context, len(episodes), self.counter.count
        )

    def _commit_episodes(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        result: Optional[SectionResult],
    ) -> None:
        """Append the episode block when the budget can absorb it.

        The selector inside ``get_episodes_for_context`` already capped by
        ``budget.episodes``; the ``budget.use`` gate here is a defensive
        check against pool exhaustion (codex round 1 #1).
        """
        if result is None:
            return
        if budget.use(result.name, result.tokens, items=result.items):
            assembly.system_prompt = (
                f"{assembly.system_prompt}\n\n{result.append_text}"
            )
            assembly.episode_count = result.items
            result.committed = True
            logger.debug(f"Added {assembly.episode_count} episodes to context")
        else:
            assembly.warnings.append(
                f"episode block ({result.tokens} tokens) skipped — "
                "exceeded episodes slice plus elastic pool"
            )
            logger.warning(
                "episode block skipped: %s tokens did not fit episodes "
                "slice plus pool",
                result.tokens,
            )

    async def _produce_memories(
        self,
        budget: TokenBudget,
        query: str,
        emotional_context: Optional[Dict[str, Any]],
        retrieval_cfg: Dict[str, float],
        *,
        include_memories: bool,
        trivial_turn: bool,
    ) -> Optional[SectionResult]:
        """Retrieve emotionally-weighted memories for dynamic user context."""
        if not (include_memories and self.memory_retriever and not trivial_turn):
            return None
        try:
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                max_tokens=budget.memories,
                counter=self.counter,
                emotional_context=emotional_context,
                min_score=retrieval_cfg["memory_min_score"],
                min_relevance=retrieval_cfg["memory_min_relevance"],
                read_only=True,
                return_details=True,
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return SectionResult(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                warning=f"Memory retrieval unavailable: {e}",
            )
        if not memories:
            return None
        from kestrel_sovereign.agent.memory_manager import RetrievedMemoryBlock

        is_block = isinstance(memories, RetrievedMemoryBlock)
        memory_text = memories.text if is_block else memories
        # Shared with ``measure_context_breakdown`` via ``context_stages``
        # so the raw-block token cost, ``[Memory]`` count, and ``<memories>``
        # wrapping are single-sourced across the two assemblers.
        return build_memory_section(
            memory_text,
            self.counter.count,
            message_ids=tuple(memories.message_ids) if is_block else (),
            is_memory_block=is_block,
        )

    async def _produce_rag(
        self,
        budget: TokenBudget,
        query: str,
        retrieval_cfg: Dict[str, float],
        *,
        include_rag: bool,
        trivial_turn: bool,
    ) -> Optional[SectionResult]:
        """Retrieve RAG documents for dynamic user context."""
        if not (include_rag and not trivial_turn):
            return None
        try:
            rag_context = await self.context_builder.retrieve_context(
                query, min_score=retrieval_cfg["rag_min_score"],
            )
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return SectionResult(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                warning=f"Document search unavailable: {e}",
            )
        if not rag_context:
            return None
        # Shared with ``measure_context_breakdown`` via ``context_stages``
        # so the raw-block token cost, chunk count, and ``<documents>``
        # wrapping are single-sourced across the two assemblers.
        return build_rag_section(rag_context, self.counter.count)

    def _commit_dynamic_section(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        result: Optional[SectionResult],
    ) -> None:
        """Commit a DYNAMIC section into the retrieved-context block.

        Budget is charged on the RAW block cost (existing semantics) while
        the WRAPPED block is what lands in dynamic user context. A retrieval
        error surfaces as a warning and contributes nothing.
        """
        if result is None:
            return
        if result.warning:
            assembly.warnings.append(result.warning)
            return
        if budget.can_fit(result.name, result.tokens):
            budget.use(result.name, result.tokens)
            assembly.dynamic_blocks.append(result.dynamic_block)
            if result.name == "memories":
                assembly.memory_count = result.items
                logger.debug(
                    f"Added {assembly.memory_count} memories to dynamic context"
                )
            elif result.name == "rag":
                assembly.rag_chunks = result.items
                logger.debug(
                    f"Added {assembly.rag_chunks} RAG chunks to dynamic context"
                )
            result.committed = True

    def _apply_history(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        history: List[Dict],
    ) -> None:
        """Anchor, format, and budget-reconcile the conversation history.

        Sizes against the elastic *effective* ceiling (own remaining + pool)
        so released slack from finalized sections grows the slice (codex
        round 1 #2). The lumpy anchor (#1430) uses the STATIC ``budget.history``
        ceiling so per-turn RAG/memory/episode slack variance doesn't disturb
        the anchor position (codex round 2 P2), keeping the ``messages[-2]`` /
        ``messages[-4]`` cache markers byte-stable. Then it pre-trims
        wrap-overhead overshoot (codex round 2 P1) and applies the lumpy prune
        safety net.
        """
        if hasattr(budget, "effective_budget"):
            history_max_tokens = budget.effective_budget("history")
        else:
            history_max_tokens = budget.history
        anchor = self._lumpy_anchor(history, budget.history)
        anchored_history = history[anchor:] if anchor > 0 else history
        if anchor > 0:
            logger.info(
                "lumpy anchor dropped %d oldest messages (kept %d) for "
                "cache-stable prefix",
                anchor,
                len(anchored_history),
            )
        formatted_history = self.context_builder.format_conversation_history(
            history=anchored_history,
            max_tokens=history_max_tokens,
        )
        history_tokens = self.counter.count_messages(formatted_history)
        if not budget.use("history", history_tokens, items=len(formatted_history)):
            # ``format_conversation_history`` overshot ``max_tokens`` because
            # wrap-overhead is added after its own per-message budget check,
            # and ``ElasticTokenBudget.use`` returns False WITHOUT recording,
            # so the legacy post-budget prune never sees the rejected bytes.
            # Trim oldest until the byte cost fits, then re-record.
            target = history_max_tokens
            while formatted_history and history_tokens > target:
                dropped = formatted_history.pop(0)
                dropped_tokens = (
                    self.counter.count(dropped.get("content", "") or "") + 4
                )
                history_tokens -= dropped_tokens
            assembly.warnings.append(
                f"history wrap-overhead overshot ceiling — pre-trimmed to "
                f"{len(formatted_history)} messages ({history_tokens} tokens)"
            )
            logger.warning(
                "history pre-trimmed: %s tokens, %s messages",
                history_tokens,
                len(formatted_history),
            )
            budget.use("history", history_tokens, items=len(formatted_history))

        # Check if we had to truncate significantly
        if len(formatted_history) < len(history) * 0.5:
            assembly.warnings.append(
                f"History truncated: {len(formatted_history)}/{len(history)} messages"
            )

        # Pre-send budget enforcement: if total exceeds budget, drop oldest
        # history down to ``PRUNE_TARGET_FRAC`` of the budget rather than
        # just-enough-to-fit. See ``_lumpy_prune_history``.
        if budget.total_used > budget.total_budget and len(formatted_history) > 1:
            pruned_tokens = self._lumpy_prune_history(formatted_history, budget)
            if pruned_tokens > 0:
                assembly.warnings.append(
                    f"Auto-pruned {pruned_tokens} tokens from history to fit budget"
                )
        assembly.formatted_history = formatted_history

    async def _finalize_salvage(
        self,
        assembly: ContextAssembly,
        history: List[Dict],
        *,
        mandatory_system_tokens: int,
        state_of_mind: Any,
    ) -> Optional[ContextResult]:
        """Durably salvage an identifiable persistent pruned span.

        On the feature-enabled path, an id-bearing span fails closed when its
        store is unreachable or its synchronous salvage write fails. If the
        selected history cannot be mapped to persistent row ids (for example,
        isolated in-memory history), ``compute_pruned_span`` returns ``None``
        and this conditional implementation does not establish a marker.
        """
        if not (
            is_durable_salvage_enabled()
            and len(assembly.formatted_history) < len(history)
        ):
            return None
        span = compute_pruned_span(
            history, assembly.formatted_history, self.counter.count
        )
        if span is None:
            return None
        conv_store = (
            self.conversation_manager._get_conversation_store()
            if hasattr(self.conversation_manager, "_get_conversation_store")
            else None
        )
        if conv_store is None:
            # Feature flag is on but no conv_store is available — fail closed
            # rather than silently let bytes leave the model view without a
            # durable record (codex round 1 #7).
            logger.error(
                "DEGRADED MODE: durable salvage feature flag "
                "is ON but no conversation store is reachable; "
                "the LLM call MUST NOT proceed"
            )
            assembly.warnings.append(
                "DEGRADED MODE: durable salvage feature flag "
                "is enabled but the conversation store is not "
                "reachable. The LLM call MUST NOT proceed."
            )
            return self._degraded_result(
                assembly,
                reason="salvage-conv-store-unavailable",
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
            )
        try:
            pending = await get_pending_count(
                conv_store, session_id=span.session_id
            )
            salvage = await salvage_messages(
                conv_store=conv_store,
                original_messages=span.dropped_messages,
                reason=SalvageReason.AUTO_PRUNE_POSTBUDGET,
                model=self.model,
                session_id=span.session_id,
                token_estimate=span.token_estimate,
                pending_count=pending,
            )
            worker = getattr(self, "salvage_worker", None)
            if worker is not None and not salvage.pointer_only_terminal:
                worker.schedule_summary(salvage.salvage_id)
            assembly.warnings.append(
                f"context-salvage: {len(span.dropped_ids)} messages "
                f"folded into salvage marker {salvage.salvage_id} "
                f"({'pointer-only-terminal' if salvage.pointer_only_terminal else 'pointer-only — async summary scheduled'})"
            )
        except SalvageWriteError as e:
            # Fail closed (Emma 2026-05-20 hardening): the bytes would leave
            # the model view without a durable record, violating C's invariant.
            logger.error(
                "DEGRADED MODE: salvage write failed (%s); "
                "LLM call MUST NOT proceed",
                e,
            )
            assembly.warnings.append(
                f"DEGRADED MODE: durable salvage write failed "
                f"({e}). The LLM call MUST NOT proceed — "
                "see logs and consider !context restore."
            )
            return self._degraded_result(
                assembly,
                reason=f"salvage-write-failed: {e}",
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
            )
        return None

    def _degraded_result(
        self,
        assembly: ContextAssembly,
        *,
        reason: str,
        mandatory_system_tokens: int,
        state_of_mind: Any,
    ) -> ContextResult:
        """Build the empty, degraded-mode ContextResult (fail-closed).

        Mirrors the degraded shape used by both the mandatory-floor and the
        salvage fail-closed paths so the caller cannot drift them apart.
        """
        return ContextResult(
            system_prompt="",
            messages=[],
            total_tokens=0,
            budget_summary={"mode": "degraded", "reason": reason},
            warnings=assembly.warnings,
            degraded_mode=True,
            mandatory_system_tokens=mandatory_system_tokens,
            state_of_mind=state_of_mind,
        )

    def _resolve_state_of_mind_snapshot(self) -> Tuple[Any, Any]:
        """Resolve the state and prompt adaptation exactly once per build."""
        if not self.llm_service or not hasattr(
            self.llm_service, "get_state_of_mind"
        ):
            return None, None
        try:
            state_of_mind = self.llm_service.get_state_of_mind()
            return state_of_mind, state_of_mind.prompt_adaptation
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to get constitutional state of mind: %s", exc
            )
            return None, None

    async def _build_ephemeral_context(
        self,
        query: str,
        constitution: str,
        include_briefing: bool,
        system_prompt_addendum: Optional[str] = None,
        system_prompt_budget_bytes: Optional[int] = None,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
        state_of_mind: Any = None,
        prompt_adaptation: Any = None,
    ) -> ContextResult:
        """
        Build minimal context for EPHEMERAL privacy mode.

        In EPHEMERAL mode, no history or memories are retrieved.
        Only the system prompt and constitution are included.
        """
        # The EPHEMERAL MODE notice is fixed text appended after the
        # budget-aware assembly. Codex round-13 P2 caught that we
        # must reserve its bytes too, otherwise the notice can push
        # the final prompt over the configured budget. Shared with the
        # append below via the ``EPHEMERAL_NOTICE`` constant so the
        # reserved and appended bytes cannot drift.
        ephemeral_notice = EPHEMERAL_NOTICE

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
                state_of_mind=None,
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
                state_of_mind=None,
                system_prompt_addendum=system_prompt_addendum,
            )

        # Append the ephemeral notice (already accounted for in the
        # reserved budget above when budget_bytes was set).
        system_prompt = f"{system_prompt}\n\n{EPHEMERAL_NOTICE}"

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
            state_of_mind=state_of_mind,
        )

    # Delegate to ConversationManager
    def _get_conversation_store(self):
        """Resolve the conversation store via the ConversationManager.

        Mirrors the internal delegation pattern already used for the
        salvage worker (see start_salvage_worker) so callers that hold a
        ContextManager can reach the same store the ConversationManager
        uses. Returns None when no store is available.
        """
        getter = getattr(
            self.conversation_manager, "_get_conversation_store", None
        )
        if getter is None:
            return None
        return getter()

    async def compact_session(self, llm_service, preserve_recent: int = 10, force: bool = False, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.compact_session(
            llm_service, self.counter, preserve_recent, force, session_id=session_id
        )

    async def check_compaction_needed(self, utilization_threshold: float = 70.0) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.check_compaction_needed(
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

    async def hierarchical_compact(self, llm_service, chunk_size: int = 4000, preserve_recent: int = 5, max_depth: int = 3) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.hierarchical_compact(
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

    # Lumpy prune: when history overflows the budget, drop down to this
    # fraction of the budget instead of just-enough-to-fit. The headroom
    # buys multiple cache-warm turns between prune events; just-enough
    # would prune ~one turn's tokens every turn and invalidate the
    # ``messages[-2]``/``[-4]`` cache markers on every request (see
    # ``project_anthropic_cache_markers.md``). Override with
    # ``KESTREL_PRUNE_TARGET_FRAC``; bad/out-of-range values fall back
    # to the default rather than crashing the import.
    @staticmethod
    def _resolve_prune_target_frac(raw: Optional[str], default: float = 0.75) -> float:
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "KESTREL_PRUNE_TARGET_FRAC=%r is not a number — using default %.2f",
                raw,
                default,
            )
            return default
        return max(0.05, min(1.0, value))

    PRUNE_TARGET_FRAC = _resolve_prune_target_frac.__func__(
        os.environ.get("KESTREL_PRUNE_TARGET_FRAC")
    )

    @staticmethod
    def _emit_content_for_msg(msg: Dict[str, Any]) -> str:
        """Select the emit bytes for a message (see
        :func:`context_stages.emit_content_for_msg`).

        Mirrors ``ContextBuilder.format_conversation_history`` so the anchor
        counts the SAME bytes the LLM will see. Retained as a method for the
        lumpy-anchor tests that call it directly.
        """
        from kestrel_sovereign.security.input_guardrails import wrap_user_input

        return emit_content_for_msg(msg, wrap_user_input)

    def _lumpy_anchor(
        self,
        history: List[Dict[str, Any]],
        max_tokens: int,
    ) -> int:
        """Compute the oldest-message index to KEEP for a cache-stable
        history window (see :func:`context_stages.compute_lumpy_anchor`).

        Counts use ``_emit_content_for_msg`` to match the bytes
        ``format_conversation_history`` will emit (including sent-form
        rendered content and ``wrap_user_input`` expansion) so the
        formatter's just-enough skip path can't run inside the anchored
        slice and undo the hysteresis.
        """
        return compute_lumpy_anchor(
            history,
            max_tokens,
            prune_target_frac=self.PRUNE_TARGET_FRAC,
            count_msg_tokens=lambda m: self.counter.count(
                self._emit_content_for_msg(m)
            ),
        )

    def _lumpy_prune_history(
        self,
        formatted_history: List[Dict[str, Any]],
        budget: TokenBudget,
    ) -> int:
        """Drop oldest history down to ``PRUNE_TARGET_FRAC`` of the budget.

        Defensive safety net for the rare case where total budget
        accounting overshoots after all sections have sized (e.g. when
        ``ElasticTokenBudget.use`` returns False without recording). The
        primary cache-stable path is ``_lumpy_anchor`` running before
        ``format_conversation_history``.

        Mutates ``formatted_history`` in place, updates ``budget``'s
        ``history`` allocation, and returns the total tokens dropped.
        Will drain the list if necessary; the current user turn is
        appended downstream by the caller, not held in this list.
        """
        overage = budget.total_used - budget.total_budget
        if overage <= 0 or not formatted_history:
            return 0
        target_total = int(budget.total_budget * self.PRUNE_TARGET_FRAC)
        target_drop = max(overage, budget.total_used - target_total)
        logger.warning(
            "Context budget exceeded by %d tokens — lumpy-pruning to "
            "%.0f%% of budget (target drop ~%d tokens)",
            overage,
            self.PRUNE_TARGET_FRAC * 100,
            target_drop,
        )
        pruned_tokens = 0
        while formatted_history and pruned_tokens < target_drop:
            dropped = formatted_history.pop(0)
            dropped_tokens = self.counter.count(dropped.get("content", "") or "") + 4
            pruned_tokens += dropped_tokens
        new_history_tokens = self.counter.count_messages(formatted_history)
        alloc = budget.allocations["history"]
        alloc.used = new_history_tokens
        alloc.items = len(formatted_history)
        return pruned_tokens

    # --- Microcompact: zero-cost tool result clearing ---

    MICROCOMPACT_KEEP_RECENT = int(os.environ.get("KESTREL_MICROCOMPACT_KEEP_RECENT", "5"))

    def _microcompact_tool_results(self, history: List[Dict]) -> int:
        """Clear stale tool-result content from conversation history in place.

        Delegates to :func:`context_stages.microcompact_tool_results`;
        retained as a method so callers/tests can reach it via the
        ``ContextManager`` instance with its ``MICROCOMPACT_KEEP_RECENT``
        policy.
        """
        return microcompact_tool_results(history, self.MICROCOMPACT_KEEP_RECENT)
