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
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

from .context_builder import ContextBuilder, _count_tool_schema_tokens
from .context_stages import (
    ContextAssembly,
    ContextBuildMode,
    ContextBuildPlan,
    ContextSectionPlan,
    SectionDestination,
    SectionResult,
    SectionStatus,
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
from .system_prompt_assembler import MandatorySystemPromptBudgetError
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

# The model-bound turn paths and read-only status acquisition must read the
# same history window before handing it to the canonical planner.
CONTEXT_HISTORY_LIMIT = 50


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
    # Constitutional-injection tracking — populated whenever the
    # priority-aware assembler runs, including turns with lifecycle-owned
    # feature context (kestrel-sovereign#1137 and #3025).
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
    # Exact canonical assertions rendered into this turn's dynamic context.
    # This is a content-free, immutable identity projection from the committed
    # plan — never a mutable ContextBuilder ``last_*`` side channel.  Turn
    # persistence uses it to make later fact erasure retract only artifacts
    # that actually depended on the assertion.
    semantic_recall_dependencies: Tuple[Tuple[str, str], ...] = ()


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
            # No ``session_id`` (#2940): the salvage janitor drains rows
            # queued by earlier, already-finished turns, so there is no chat
            # window this call belongs to.
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
        tools: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> ContextResult:
        """Plan, commit, and render the production context for one turn."""

        plan = await self.build_context_plan(
            query=query,
            constitution=constitution,
            include_briefing=include_briefing,
            include_memories=include_memories,
            include_rag=include_rag,
            privacy_mode=privacy_mode,
            emotional_context=emotional_context,
            conversation_history=conversation_history,
            reflection_guidance=reflection_guidance,
            system_prompt_addendum=system_prompt_addendum,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
            anchored_doctrine=anchored_doctrine,
            tools=tools,
            mode=ContextBuildMode.LIVE,
            session_id=session_id,
        )
        return await self._commit_and_render_plan(plan)

    async def build_context_plan(
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
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        mode: ContextBuildMode = ContextBuildMode.DRY_RUN,
        measure_expensive_sections: bool = True,
        message_count_override: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> ContextBuildPlan:
        """
        Build the canonical read-only plan for an LLM request.

        Both production and context-status call this method.  It executes the
        same relevance gates, elastic-budget finalization, lumpy anchor,
        microcompaction, wrapper accounting, and final prune decisions in both
        modes.  It never writes access rehearsal or salvage rows.  A live
        caller must commit the plan through :meth:`_commit_and_render_plan`.

        It:
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
            session_id: Chat session of the turn being built, for span
                attribution only (#2940). Memory retrieval runs an
                answerability judge through the LLM; naming the session keeps
                that call in the turn's Timeline band instead of a band of its
                own. It selects nothing — history filtering is still the
                caller's job via ``conversation_history``. The context-status
                poll and other dry-run callers have no turn and pass nothing.

        ``measure_expensive_sections=False`` preserves the cheap footer poll:
        memory/RAG rows are marked ``unknown`` or ``skipped`` with
        ``tokens=None`` rather than reported as measured zero.

        Returns:
            Typed read-only plan with exact artifacts and required side effects.
        """
        # One typed state object per call — never a shared instance — so
        # concurrent COGNITION dispatches cannot cross-contaminate counters
        # or results. Per-task injection tracking is published via the
        # ContextVar only after a live plan commits successfully.
        assembly = ContextAssembly()
        sections: Dict[str, ContextSectionPlan] = {}

        # Resolve constitutional awareness once. Prompt adaptation is stable
        # top-level context; mutable StateOfMind fields travel as append-only
        # operator facts. Returning this exact object in ContextResult keeps
        # those two views on the same per-turn snapshot.
        state_of_mind, prompt_adaptation = (
            self._resolve_state_of_mind_snapshot()
        )

        # Handle EPHEMERAL mode - no retrieval
        if privacy_mode == "EPHEMERAL":
            return await self._build_ephemeral_plan(
                query=query,
                constitution=constitution,
                include_briefing=include_briefing,
                system_prompt_addendum=system_prompt_addendum,
                system_prompt_budget_bytes=system_prompt_budget_bytes,
                anchored_doctrine=anchored_doctrine,
                state_of_mind=state_of_mind,
                prompt_adaptation=prompt_adaptation,
                mode=mode,
                tools=tools,
            )

        # Use provided history or fetch from storage
        if conversation_history is not None:
            history = deepcopy(conversation_history)
        else:
            history = deepcopy(
                await self.conversation_manager.get_conversation_history()
            )
        message_count = (
            len(history)
            if message_count_override is None
            else message_count_override
        )

        # Tool schemas share the provider payload window with every context
        # section. Measure them before admitting optional system clauses so a
        # large direct-tool surface cannot make an otherwise-valid system
        # slice overflow the actual request ceiling.
        tools_tokens = _count_tool_schema_tokens(self.counter, tools)

        # Fund the exact formatter this turn will send. A zero-contribution
        # legacy render keeps its historical bytes, but when those optional
        # bytes overflow the legacy-funded system slice the priority assembler
        # becomes the selected formatter and its (slightly different) mandatory
        # wrapper must fund the elastic floor too.
        requires_tracking = self._requires_tracked_system_prompt(
            system_prompt_budget_bytes=system_prompt_budget_bytes,
            anchored_doctrine=anchored_doctrine,
        )
        legacy_system_render = None

        def tool_aware_budget(mandatory_floor: int):
            candidate = create_budget(
                self.model,
                message_count,
                elastic=True,
                mandatory_system_tokens=mandatory_floor,
            )
            if mandatory_floor + tools_tokens > candidate.total_budget:
                raise DegradedModeError(
                    mandatory_floor,
                    candidate.total_budget,
                    self.model,
                    detail=(
                        "mandatory governance floor and tool schemas do not fit "
                        f"the model payload budget ({mandatory_floor} + "
                        f"{tools_tokens} > {candidate.total_budget} tokens)"
                    ),
                )
            candidate.reserve_external(tools_tokens)
            return candidate

        mandatory_system_tokens = self._measure_mandatory_system_tokens(
            constitution,
            prompt_adaptation,
            anchored_doctrine=anchored_doctrine,
            required_suffix=system_prompt_addendum,
            tracked_prompt=requires_tracking,
        )
        try:
            budget = tool_aware_budget(mandatory_system_tokens)
            if not requires_tracking:
                legacy_system_render = (
                    self.context_builder.build_system_prompt_with_subsections(
                        constitution=constitution,
                        include_briefing=include_briefing,
                        prompt_adaptation=prompt_adaptation,
                        state_of_mind=None,
                        system_prompt_addendum=system_prompt_addendum,
                    )
                )
                if (
                    self.counter.count(legacy_system_render[0])
                    > budget.allocations["system"].budget
                ):
                    requires_tracking = True
                    legacy_system_render = None
                    mandatory_system_tokens = (
                        self._measure_mandatory_system_tokens(
                            constitution,
                            prompt_adaptation,
                            anchored_doctrine=anchored_doctrine,
                            required_suffix=system_prompt_addendum,
                            tracked_prompt=True,
                        )
                    )
                    budget = tool_aware_budget(mandatory_system_tokens)
        except DegradedModeError as e:
            logger.error(
                "degraded-mode: %s — returning empty ContextResult; caller "
                "MUST surface this and refuse the LLM call",
                e,
            )
            assembly.warnings.append(
                f"DEGRADED MODE: {e}. The LLM call MUST NOT proceed — "
                "surface this to the operator."
            )
            return self._degraded_plan(
                assembly,
                reason=str(e),
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
                mode=mode,
            )

        tool_aware_system_budget = budget.allocations["system"].budget

        # 1. Assemble the stable system prefix (constitution/identity/doctrine)
        # and record its usage. Kept separate from the per-turn dynamic user
        # context by construction (ContextAssembly). The tracking assembler is
        # used when a per-source byte budget, anchored doctrine, or lifecycle-
        # owned context is present; otherwise the byte-stable legacy assembler.
        try:
            (
                assembly.system_prompt,
                assembly.injected_clauses,
                assembly.dropped_clauses,
                system_subsections,
            ) = self._assemble_system_prompt(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                system_prompt_addendum=system_prompt_addendum,
                system_prompt_budget_bytes=system_prompt_budget_bytes,
                system_prompt_budget_tokens=tool_aware_system_budget,
                anchored_doctrine=anchored_doctrine,
                requires_tracking=requires_tracking,
                legacy_render=legacy_system_render,
            )
        except MandatorySystemPromptBudgetError as exc:
            reason = str(exc)
            logger.error(
                "degraded-mode: %s — caller MUST refuse the LLM call",
                reason,
            )
            assembly.warnings.append(
                f"DEGRADED MODE: {reason}. The LLM call MUST NOT proceed — "
                "surface this to the operator."
            )
            return self._degraded_plan(
                assembly,
                reason=reason,
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
                mode=mode,
            )
        self._record_system_usage(assembly, budget)

        # 1b. Reflection guidance (into the system prompt, budget-gated).
        reflection_included = self._apply_reflection_guidance(
            assembly,
            budget,
            reflection_guidance,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
        )
        system_provenance = ["constitution", "bootstrap"]
        if include_briefing:
            system_provenance.append("session_briefing")
        if prompt_adaptation is not None:
            system_provenance.append("prompt_adaptation")
        if system_prompt_addendum:
            system_provenance.append("system_prompt_addendum")
        if anchored_doctrine:
            system_provenance.extend(
                f"anchored_doctrine:{name}" for name in anchored_doctrine
            )
        if reflection_included:
            system_provenance.append("reflection_guidance")
            system_subsections.append(
                (
                    "reflection_guidance",
                    build_reflection_guidance_block(reflection_guidance or []),
                )
            )
        system_tokens_before_episodes = self.counter.count(assembly.system_prompt)
        sections["system"] = ContextSectionPlan(
            name="system",
            destination=SectionDestination.SYSTEM,
            status=SectionStatus.INCLUDED,
            tokens=system_tokens_before_episodes,
            budget=budget.system,
            items=1,
            provenance=tuple(system_provenance),
            details={
                "subsections": self._measure_system_subsections(
                    system_subsections
                ),
                "injected_clauses": assembly.injected_clauses,
                "dropped_clauses": assembly.dropped_clauses,
            },
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
        if episode_result is None:
            sections["episodes"] = ContextSectionPlan(
                name="episodes",
                destination=SectionDestination.SYSTEM,
                status=SectionStatus.EMPTY,
                tokens=0,
                budget=budget.episodes,
                items=0,
                provenance=("episode_store",),
                reason=(
                    "conversation below episode threshold or no consolidator"
                ),
                details={"threshold": self.EPISODE_THRESHOLD_MESSAGES},
            )
        else:
            sections["episodes"] = ContextSectionPlan(
                name="episodes",
                destination=SectionDestination.SYSTEM,
                status=(
                    SectionStatus.INCLUDED
                    if episode_result.committed
                    else SectionStatus.EXCLUDED
                ),
                tokens=(
                    self.counter.count(episode_result.append_text or "")
                    if episode_result.committed
                    else 0
                ),
                budget=budget.episodes,
                items=episode_result.items if episode_result.committed else 0,
                provenance=("episode_store", "elastic_budget_gate"),
                reason=(
                    None
                    if episode_result.committed
                    else (
                        episode_result.warning
                        or "episode block did not fit the elastic budget"
                    )
                ),
                raw_tokens=episode_result.tokens,
                details={"threshold": self.EPISODE_THRESHOLD_MESSAGES},
            )
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
        memory_result: Optional[SectionResult] = None
        if measure_expensive_sections:
            memory_result = await self._produce_memories(
                budget,
                query,
                emotional_context,
                retrieval_cfg,
                include_memories=include_memories,
                trivial_turn=trivial_turn,
                session_id=session_id,
            )
        self._commit_dynamic_section(assembly, budget, memory_result)
        memory_access_ids: Tuple[int, ...] = ()
        if memory_result is not None and memory_result.committed:
            if memory_result.is_memory_block:
                memory_access_ids = memory_result.message_ids
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.INCLUDED,
                tokens=self.counter.count(memory_result.dynamic_block or ""),
                budget=budget.memories,
                items=memory_result.items,
                provenance=(
                    "memory_retriever",
                    "query_relevance_gate",
                    "elastic_budget_gate",
                ),
                raw_tokens=memory_result.tokens,
                details={"wired": self.memory_retriever is not None},
            )
        elif not include_memories or trivial_turn:
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.SKIPPED,
                tokens=None,
                budget=budget.memories,
                items=None,
                provenance=("query_relevance_gate",),
                reason=(
                    "excluded by turn relevance gate"
                    if trivial_turn
                    else "memory inclusion disabled"
                ),
                details={"wired": self.memory_retriever is not None},
            )
        elif not measure_expensive_sections:
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.UNKNOWN,
                tokens=None,
                budget=budget.memories,
                items=None,
                provenance=("measurement_policy",),
                reason="not acquired on the cheap status path",
                details={"wired": self.memory_retriever is not None},
            )
        elif memory_result is not None and not memory_result.warning:
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.EXCLUDED,
                tokens=0,
                budget=budget.memories,
                items=0,
                provenance=("memory_retriever", "elastic_budget_gate"),
                reason="memory block did not fit the elastic budget",
                raw_tokens=memory_result.tokens,
                details={"wired": self.memory_retriever is not None},
            )
        elif memory_result is not None and memory_result.warning:
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.ERROR,
                tokens=None,
                budget=budget.memories,
                items=None,
                provenance=("memory_retriever",),
                reason=memory_result.warning,
                details={"wired": self.memory_retriever is not None},
            )
        else:
            sections["memories"] = ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.EMPTY,
                tokens=0,
                budget=budget.memories,
                items=0,
                provenance=("memory_retriever", "query_relevance_gate"),
                reason=(
                    "no memory retriever configured"
                    if self.memory_retriever is None
                    else "no relevant memories"
                ),
                details={"wired": self.memory_retriever is not None},
            )
        if (
            measure_expensive_sections
            or not include_memories
            or trivial_turn
        ):
            finalize_section(budget, "memories")

        # 4. RAG documents (into dynamic user context, not system).
        rag_result: Optional[SectionResult] = None
        if measure_expensive_sections:
            rag_result = await self._produce_rag(
                budget,
                query,
                retrieval_cfg,
                include_rag=include_rag,
                trivial_turn=trivial_turn,
                session_id=session_id,
            )
        self._commit_dynamic_section(assembly, budget, rag_result)
        if rag_result is not None and rag_result.committed:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.INCLUDED,
                tokens=self.counter.count(rag_result.dynamic_block or ""),
                budget=budget.rag,
                items=rag_result.items,
                provenance=(
                    "rag_store",
                    "query_relevance_gate",
                    "elastic_budget_gate",
                ),
                raw_tokens=rag_result.tokens,
                details={
                    "chunks": rag_result.items,
                    "estimated": True,
                    "estimation_method": "production-retrieval-plan",
                    **rag_result.metadata,
                },
            )
        elif not include_rag or trivial_turn:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.SKIPPED,
                tokens=None,
                budget=budget.rag,
                items=None,
                provenance=("query_relevance_gate",),
                reason=(
                    "excluded by turn relevance gate"
                    if trivial_turn
                    else "RAG inclusion disabled"
                ),
                details={"chunks": None, "skipped": True},
            )
        elif not measure_expensive_sections:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.UNKNOWN,
                tokens=None,
                budget=budget.rag,
                items=None,
                provenance=("measurement_policy",),
                reason="not acquired on the cheap status path",
                details={"chunks": None, "skipped": True},
            )
        elif rag_result is not None and not rag_result.warning:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.EXCLUDED,
                tokens=0,
                budget=budget.rag,
                items=0,
                provenance=("rag_store", "elastic_budget_gate"),
                reason="RAG block did not fit the elastic budget",
                raw_tokens=rag_result.tokens,
                details={"chunks": 0, "skipped": False},
            )
        elif rag_result is not None and rag_result.warning:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.ERROR,
                tokens=None,
                budget=budget.rag,
                items=None,
                provenance=("rag_store",),
                reason=rag_result.warning,
                details={"chunks": None, "skipped": False},
            )
        else:
            sections["rag"] = ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.EMPTY,
                tokens=0,
                budget=budget.rag,
                items=0,
                provenance=("rag_store", "query_relevance_gate"),
                reason="no relevant documents",
                details={"chunks": 0, "skipped": False},
            )
        if measure_expensive_sections or not include_rag or trivial_turn:
            finalize_section(budget, "rag")

        # 5. Format conversation history with the remaining (elastic) budget,
        # lumpy-anchored for a cache-stable prefix, then reconciled to fit.
        self._apply_history(assembly, budget, history)
        rendered_payload_tokens = self._final_prune_to_payload_budget(
            assembly, budget, extra_tokens=tools_tokens
        )
        if rendered_payload_tokens > budget.total_budget:
            reason = (
                "rendered non-history context does not fit the model payload "
                f"budget ({rendered_payload_tokens} > {budget.total_budget} tokens)"
            )
            assembly.warnings.append(
                f"DEGRADED MODE: {reason}. The LLM call MUST NOT proceed — "
                "surface this to the operator."
            )
            assembly.system_prompt = ""
            assembly.dynamic_blocks.clear()
            assembly.formatted_history.clear()
            return self._degraded_plan(
                assembly,
                reason=reason,
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
                mode=mode,
            )

        raw_history_tokens = sum(
            self.counter.count(m.get("content", "") or "") for m in history
        )
        history_tokens = self.counter.count_messages(assembly.formatted_history)
        sections["history"] = ContextSectionPlan(
            name="history",
            destination=SectionDestination.HISTORY,
            status=(
                SectionStatus.INCLUDED
                if assembly.formatted_history
                else SectionStatus.EMPTY
            ),
            tokens=history_tokens,
            budget=budget.effective_budget("history")
            if hasattr(budget, "effective_budget")
            else budget.history,
            items=len(assembly.formatted_history),
            provenance=("conversation_history", "lumpy_anchor", "final_prune"),
            raw_tokens=raw_history_tokens,
            details={
                "messages_total": len(history),
                "messages_kept_after_pruning": len(assembly.formatted_history),
            },
        )

        durable_salvage_enabled = is_durable_salvage_enabled()
        pruned_span = None
        salvage_requirement = None
        if len(assembly.formatted_history) < len(history):
            pruned_span = compute_pruned_span(
                history, assembly.formatted_history, self.counter.count
            )
            if (
                durable_salvage_enabled
                and pruned_span is not None
                and pruned_span.dropped_ids
            ):
                salvage_requirement = pruned_span
                assembly.warnings.append(
                    "context-salvage commit required before the LLM call: "
                    f"{len(salvage_requirement.dropped_ids)} messages "
                    "must be synchronously recorded"
                )
            if (
                durable_salvage_enabled
                and pruned_span is not None
                and pruned_span.unmappable_count
            ):
                assembly.warnings.append(
                    "context-salvage cannot durably link "
                    f"{pruned_span.unmappable_count} pruned in-memory/id-less "
                    "messages; those omissions remain silently pruned"
                )

        sections["tools"] = ContextSectionPlan(
            name="tools",
            destination=SectionDestination.TOOLS,
            status=(
                SectionStatus.INCLUDED if tools else SectionStatus.EMPTY
            ),
            tokens=tools_tokens,
            items=len(tools or []),
            provenance=("tool_registry", "json_serialized_schemas"),
            details={
                "estimated": True,
                "estimation_method": "json-serialized-schemas",
            },
        )
        dynamic_tokens = self.counter.count(assembly.dynamic_user_context)
        inner_dynamic_tokens = sum(
            section.tokens or 0
            for name, section in sections.items()
            if name in {"memories", "rag"} and section.included
        )
        sections["dynamic_context_overhead"] = ContextSectionPlan(
            name="dynamic_context_overhead",
            destination=SectionDestination.DYNAMIC,
            status=(
                SectionStatus.INCLUDED
                if assembly.dynamic_user_context
                else SectionStatus.EMPTY
            ),
            tokens=max(0, dynamic_tokens - inner_dynamic_tokens),
            items=1 if assembly.dynamic_user_context else 0,
            provenance=("retrieved_context_wrapper",),
            details={
                "applies_when": "memories or rag included",
                "applied": bool(assembly.dynamic_user_context),
            },
        )

        # Attribute the episode joiner's tokenizer effect to the base system
        # row so the section sum equals the exact rendered system bytes.
        sections["system"].tokens = max(
            0,
            self.counter.count(assembly.system_prompt)
            - (sections["episodes"].tokens or 0),
        )
        subsection_rows = sections["system"].details["subsections"]
        attributed = sum(row["tokens"] for row in subsection_rows)
        if subsection_rows:
            # Episodes are accounted as their own section. Assign only the
            # tokenizer boundary delta from their late append to the final
            # base-system subsection so all attribution remains exact.
            subsection_rows[-1]["tokens"] += (
                sections["system"].tokens - attributed
            )
        total_tokens = (
            self.counter.count(assembly.system_prompt)
            + history_tokens
            + dynamic_tokens
            + tools_tokens
        )

        logger.info(
            f"Context planned: {total_tokens}/{budget.total_budget} tokens "
            f"({len(assembly.formatted_history)} msgs, {assembly.episode_count} episodes, "
            f"{assembly.memory_count} memories, {assembly.rag_chunks} docs)"
        )

        return ContextBuildPlan(
            mode=mode,
            model=self.model,
            assembly=assembly,
            sections=sections,
            budget_summary=budget.get_summary(),
            context_limit=budget.context_limit,
            response_reserve=budget.response_reserve,
            total_budget=budget.total_budget,
            total_tokens=total_tokens,
            mandatory_system_tokens=mandatory_system_tokens,
            state_of_mind=state_of_mind,
            memory_access_ids=memory_access_ids,
            salvage_requirement=salvage_requirement,
            pruned_span=pruned_span,
            durable_salvage_enabled=durable_salvage_enabled,
            measurement_complete=not any(
                section.status is SectionStatus.UNKNOWN
                for section in sections.values()
            ),
            microcompacted_tool_results=microcompact_savings,
        )

    async def _commit_and_render_plan(
        self, plan: ContextBuildPlan
    ) -> ContextResult:
        """Commit a live plan's required writes, then render its artifacts."""

        if plan.degraded_mode:
            return self._render_context_plan(plan)

        if plan.memory_access_ids and self.memory_retriever is not None:
            try:
                await self.memory_retriever.record_accesses(
                    plan.memory_access_ids, self.agent_id
                )
            except Exception as exc:
                logger.warning("Memory access bookkeeping failed: %s", exc)
                plan.warnings.append(
                    f"Memory access bookkeeping unavailable: {exc}"
                )

        span = plan.salvage_requirement
        if span is not None:
            degraded = await self._commit_salvage_requirement(plan, span)
            if degraded is not None:
                return degraded

        # Publish only after every fail-closed side effect has committed.
        _INJECTION_TRACKING_VAR.set(
            (
                plan.assembly.injected_clauses,
                plan.assembly.dropped_clauses,
            )
        )
        return self._render_context_plan(plan)

    async def _commit_salvage_requirement(
        self,
        plan: ContextBuildPlan,
        span,
    ) -> Optional[ContextResult]:
        """Synchronously commit the durable salvage declared by ``plan``."""

        conv_store = self._get_conversation_store()
        if conv_store is None:
            logger.error(
                "DEGRADED MODE: durable salvage is required but no "
                "conversation store is reachable"
            )
            plan.warnings.append(
                "DEGRADED MODE: durable salvage feature is enabled but the "
                "conversation store is not reachable. The LLM call MUST NOT "
                "proceed."
            )
            return self._degraded_result(
                plan.assembly,
                reason="salvage-conv-store-unavailable",
                mandatory_system_tokens=plan.mandatory_system_tokens,
                state_of_mind=plan.state_of_mind,
            )

        try:
            pending = await get_pending_count(
                conv_store, session_id=span.session_id
            )
            salvage = await salvage_messages(
                conv_store=conv_store,
                original_messages=span.dropped_messages,
                reason=SalvageReason.AUTO_PRUNE_POSTBUDGET,
                model=plan.model,
                session_id=span.session_id,
                token_estimate=span.token_estimate,
                pending_count=pending,
            )
            worker = getattr(self, "salvage_worker", None)
            if worker is not None and not salvage.pointer_only_terminal:
                worker.schedule_summary(salvage.salvage_id)
            plan.warnings.append(
                f"context-salvage: {len(span.dropped_ids)} messages folded "
                f"into salvage marker {salvage.salvage_id} "
                f"({'pointer-only-terminal' if salvage.pointer_only_terminal else 'pointer-only — async summary scheduled'})"
            )
        except SalvageWriteError as exc:
            logger.error(
                "DEGRADED MODE: salvage write failed (%s); "
                "LLM call MUST NOT proceed",
                exc,
            )
            plan.warnings.append(
                f"DEGRADED MODE: durable salvage write failed ({exc}). "
                "The LLM call MUST NOT proceed — see logs and consider "
                "!context restore."
            )
            return self._degraded_result(
                plan.assembly,
                reason=f"salvage-write-failed: {exc}",
                mandatory_system_tokens=plan.mandatory_system_tokens,
                state_of_mind=plan.state_of_mind,
            )
        return None

    @staticmethod
    def _render_context_plan(plan: ContextBuildPlan) -> ContextResult:
        """Render a committed live plan without re-running policy."""

        assembly = plan.assembly
        return ContextResult(
            system_prompt=assembly.system_prompt,
            messages=assembly.formatted_history,
            total_tokens=plan.total_tokens,
            budget_summary=plan.budget_summary,
            episode_count=assembly.episode_count,
            memory_count=assembly.memory_count,
            rag_chunks=assembly.rag_chunks,
            warnings=assembly.warnings,
            dynamic_user_context=assembly.dynamic_user_context,
            injected_clauses=assembly.injected_clauses,
            dropped_clauses=assembly.dropped_clauses,
            degraded_mode=plan.degraded_mode,
            mandatory_system_tokens=plan.mandatory_system_tokens,
            state_of_mind=plan.state_of_mind,
            semantic_recall_dependencies=ContextManager._semantic_recall_dependencies(
                plan
            ),
        )

    @staticmethod
    def _semantic_recall_dependencies(
        plan: ContextBuildPlan,
    ) -> Tuple[Tuple[str, str], ...]:
        """Return assertion/revision identities from the committed RAG plan.

        Only an INCLUDED RAG section is model-visible.  The values below are
        deliberately extracted from the typed section details that fed that
        section, not from ``ContextBuilder.last_semantic_recall_metadata``:
        another concurrent context build may already have overwritten that
        mutable diagnostic field by the time the caller persists this turn.
        """
        rag = plan.sections.get("rag")
        if rag is None or not rag.included:
            return ()
        recall = rag.details.get("semantic_recall")
        if not isinstance(recall, dict) or recall.get("status") != "used":
            return ()
        assertions = recall.get("assertions")
        if not isinstance(assertions, (list, tuple)):
            return ()

        dependencies: list[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            assertion_id = assertion.get("assertion_id")
            revision_id = assertion.get("revision_id")
            if (
                not isinstance(assertion_id, str)
                or not assertion_id
                or not isinstance(revision_id, str)
                or not revision_id
            ):
                continue
            dependency = (assertion_id, revision_id)
            if dependency not in seen:
                seen.add(dependency)
                dependencies.append(dependency)
        return tuple(dependencies)

    # ------------------------------------------------------------------
    # Canonical plan stages — each produces/commits one section. Content
    # vocabulary lives in ``context_stages``;
    # the elastic finalization boundary is applied by the orchestrator, not
    # by these stages (they never call ``mark_section_finalized``).
    # ------------------------------------------------------------------

    def _measure_mandatory_system_tokens(
        self,
        constitution: str,
        prompt_adaptation: Any,
        *,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
        required_suffix: Optional[str] = None,
        tracked_prompt: bool = False,
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
            anchored_doctrine=anchored_doctrine,
            required_suffix=required_suffix,
            tracked_prompt=tracked_prompt,
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

    def _requires_tracked_system_prompt(
        self,
        *,
        system_prompt_budget_bytes: Optional[int],
        anchored_doctrine: Optional["OrderedDict[str, str]"],
    ) -> bool:
        """Return whether policy inputs require the priority-aware formatter."""

        context_clause_probe = getattr(
            type(self.context_builder), "has_context_clauses", None
        )
        has_context_clauses = bool(
            context_clause_probe(self.context_builder)
            if callable(context_clause_probe)
            else False
        )
        return bool(
            system_prompt_budget_bytes is not None
            or anchored_doctrine
            or has_context_clauses
        )

    def _assemble_system_prompt(
        self,
        *,
        constitution: str,
        include_briefing: bool,
        prompt_adaptation: Any,
        system_prompt_addendum: Optional[str],
        system_prompt_budget_bytes: Optional[int],
        system_prompt_budget_tokens: int,
        anchored_doctrine: Optional["OrderedDict[str, str]"],
        requires_tracking: bool,
        legacy_render: Optional[Tuple[str, List[Tuple[str, str]]]],
    ) -> Tuple[
        str,
        Optional[List[str]],
        Optional[List[str]],
        List[Tuple[str, str]],
    ]:
        """Assemble the stable system prefix and optional injection tracking.

        Routes to the priority-aware tracking assembler when the caller sets
        a per-source byte budget, supplies anchored doctrine, or has active
        lifecycle-owned context. The legacy path remains byte-identical for
        callers with none of those inputs. Contributed clauses are additionally
        bounded by the system allocation's exact token ceiling, with the
        addendum measured as a required non-droppable suffix.

        Returns ``(system_prompt, injected_clauses, dropped_clauses,
        subsections)``; the clause lists are ``None`` for the legacy path.
        """
        injected_clauses: Optional[List[str]] = None
        dropped_clauses: Optional[List[str]] = None
        subsections: List[Tuple[str, str]]
        if not requires_tracking:
            if legacy_render is None:
                raise RuntimeError("legacy system formatter was not pre-rendered")
            system_prompt, subsections = legacy_render
            return (
                system_prompt,
                injected_clauses,
                dropped_clauses,
                subsections,
            )

        tracking_result = self.context_builder.build_system_prompt_with_tracking(
            constitution=constitution,
            include_briefing=include_briefing,
            prompt_adaptation=prompt_adaptation,
            state_of_mind=None,
            budget_bytes=system_prompt_budget_bytes,
            budget_tokens=system_prompt_budget_tokens,
            required_suffix=system_prompt_addendum,
            anchored_doctrine=anchored_doctrine,
        )
        system_prompt = tracking_result.prompt
        injected_clauses = list(tracking_result.injected_clauses)
        dropped_clauses = list(tracking_result.dropped_clauses)
        subsections = list(tracking_result.subsections)
        if system_prompt_addendum:
            system_prompt = f"{system_prompt}\n\n{system_prompt_addendum}"
            subsections.append(
                ("system_prompt_addendum", system_prompt_addendum)
            )
        return system_prompt, injected_clauses, dropped_clauses, subsections

    def _measure_system_subsections(
        self, subsections: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Attribute exact prefix-token deltas to ordered subsection bodies."""

        rows: List[Dict[str, Any]] = []
        prefix = ""
        prior_tokens = 0
        for name, body in subsections:
            prefix = body if not prefix else f"{prefix}\n\n{body}"
            current_tokens = self.counter.count(prefix)
            rows.append(
                {"name": name, "tokens": current_tokens - prior_tokens}
            )
            prior_tokens = current_tokens
        return rows

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
    ) -> bool:
        """Append reflection guidance to the system prompt (budget-gated).

        When a per-source byte budget is in effect the late append must not
        push the total over the cap — the budget contract takes precedence
        over reflection guidance (codex round-15 P2).
        """
        if not reflection_guidance:
            return False
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
                assembly.warnings.append(
                    "reflection guidance skipped because it would exceed "
                    f"the {system_prompt_budget_bytes}-byte system prompt cap"
                )
                return False
        # Charge the exact tokenizer delta of the bytes we append, including
        # the ``\n\n`` boundary. Tokenizers may merge across that boundary, so
        # summing the body and a separately-counted joiner is not exact either.
        guidance_tokens = max(
            0,
            self.counter.count(
                f"{assembly.system_prompt}\n\n{guidance_text}"
            )
            - self.counter.count(assembly.system_prompt),
        )
        # Reflection is optional.  It must not consume bytes when the system
        # slice plus released elastic slack cannot accept the whole block.
        # Check first because the legacy TokenBudget.use mutates on rejection;
        # the production ElasticTokenBudget then provides a defensive commit
        # result as well.
        if not budget.can_fit("system", guidance_tokens) or not budget.use(
            "system", guidance_tokens
        ):
            logger.warning(
                "Skipping reflection guidance because %d tokens do not fit "
                "the remaining system allocation",
                guidance_tokens,
            )
            assembly.warnings.append(
                "reflection guidance skipped because it would exceed the "
                "remaining system token budget"
            )
            return False
        assembly.system_prompt = f"{assembly.system_prompt}\n\n{guidance_text}"
        logger.info(
            f"Injected {len(reflection_guidance)} reflection guidance items into prompt"
        )
        return True

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
                return SectionResult(
                    name="episodes",
                    destination=SectionDestination.SYSTEM,
                    tokens=self.counter.count(episode_context),
                    items=len(episodes),
                    append_text=episode_context,
                    warning=(
                        "episode context skipped because it would exceed "
                        f"the {system_prompt_budget_bytes}-byte system prompt cap"
                    ),
                )
        # Canonical typed result for episode bytes/counting.
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
        if result.warning:
            assembly.warnings.append(result.warning)
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
        session_id: Optional[str] = None,
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
                session_id=session_id,
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
        # Canonical typed result for memory bytes/counting/provenance.
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
        session_id: Optional[str] = None,
    ) -> Optional[SectionResult]:
        """Retrieve RAG documents for dynamic user context."""
        if not (include_rag and not trivial_turn):
            return None
        try:
            # ``session_id`` forwarded only when there is one, matching this
            # file's ``min_score`` idiom: the ContextBuilder is an injected
            # seam, so a sessionless build must reach it exactly as before.
            rag_kwargs: Dict[str, Any] = (
                {"session_id": session_id} if session_id else {}
            )
            rag_context = await self.context_builder.retrieve_context(
                query, min_score=retrieval_cfg["rag_min_score"], max_tokens=budget.rag,
                **rag_kwargs,
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
        # Canonical typed result for RAG bytes/counting/provenance.
        result = build_rag_section(rag_context, self.counter.count)
        # Content-free recall identifiers/scores belong in the typed section
        # trace, never in prompt text or logs containing retrieved content.
        metadata = getattr(self.context_builder, "last_semantic_recall_metadata", None)
        if isinstance(metadata, dict):
            result.metadata = {"semantic_recall": metadata}
        return result

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
        else:
            assembly.warnings.append(
                f"{result.name} block ({result.tokens} tokens) skipped — "
                f"exceeded {result.name} slice plus elastic pool"
            )

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

    def _final_prune_to_payload_budget(
        self,
        assembly: ContextAssembly,
        budget: TokenBudget,
        *,
        extra_tokens: int = 0,
    ) -> int:
        """Enforce the final rendered-byte ceiling after wrapper accounting.

        Section budgets charge the historical raw memory/RAG block costs, but
        the model receives their wrappers and the completed system prompt.
        This final boundary counts those actual bytes and lumpy-prunes oldest
        history until the rendered payload is below the configured target.
        The same method runs for live and dry-run plans.
        """

        def rendered_tokens() -> int:
            return (
                self.counter.count(assembly.system_prompt)
                + self.counter.count(assembly.dynamic_user_context)
                + self.counter.count_messages(assembly.formatted_history)
                + extra_tokens
            )

        before = rendered_tokens()
        if before <= budget.total_budget:
            return before

        target = int(budget.total_budget * self.PRUNE_TARGET_FRAC)
        dropped_tokens = 0
        dropped_messages = 0
        while assembly.formatted_history and rendered_tokens() > target:
            dropped = assembly.formatted_history.pop(0)
            dropped_tokens += (
                self.counter.count(dropped.get("content", "") or "") + 4
            )
            dropped_messages += 1

        allocation = budget.allocations["history"]
        allocation.used = self.counter.count_messages(assembly.formatted_history)
        allocation.items = len(assembly.formatted_history)
        after = rendered_tokens()
        if dropped_messages:
            assembly.warnings.append(
                "Final payload pruning removed "
                f"{dropped_messages} history messages ({dropped_tokens} tokens) "
                "after exact wrapper accounting"
            )
        if after > budget.total_budget:
            assembly.warnings.append(
                f"Rendered non-history context remains over budget "
                f"({after}/{budget.total_budget} tokens)"
            )
        return after

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

    def _degraded_plan(
        self,
        assembly: ContextAssembly,
        *,
        reason: str,
        mandatory_system_tokens: int,
        state_of_mind: Any,
        mode: ContextBuildMode,
    ) -> ContextBuildPlan:
        """Describe a mandatory-floor failure without issuing any writes."""

        context_limit = get_token_counter(self.model).get_context_limit()
        from .token_budget import RESPONSE_RESERVE

        return ContextBuildPlan(
            mode=mode,
            model=self.model,
            assembly=assembly,
            sections={},
            budget_summary={"mode": "degraded", "reason": reason},
            context_limit=context_limit,
            response_reserve=RESPONSE_RESERVE,
            total_budget=max(0, context_limit - RESPONSE_RESERVE),
            total_tokens=0,
            mandatory_system_tokens=mandatory_system_tokens,
            state_of_mind=state_of_mind,
            degraded_mode=True,
            degraded_reason=reason,
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

    async def _build_ephemeral_plan(
        self,
        query: str,
        constitution: str,
        include_briefing: bool,
        system_prompt_addendum: Optional[str] = None,
        system_prompt_budget_bytes: Optional[int] = None,
        anchored_doctrine: Optional["OrderedDict[str, str]"] = None,
        state_of_mind: Any = None,
        prompt_adaptation: Any = None,
        mode: ContextBuildMode = ContextBuildMode.DRY_RUN,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextBuildPlan:
        """
        Build the read-only minimal plan for EPHEMERAL privacy mode.

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
        from .token_budget import RESPONSE_RESERVE

        # Resolve the model limit through the canonical catalog instead of
        # assuming an injected/testing counter implements this optional query.
        # The counter itself remains the exact tokenizer used below.
        context_limit = get_token_counter(self.model).get_context_limit()
        total_budget = max(0, context_limit - RESPONSE_RESERVE)
        tools_tokens = _count_tool_schema_tokens(self.counter, tools)
        system_token_budget = max(0, total_budget - tools_tokens)
        required_suffix = "\n\n".join(
            part
            for part in (system_prompt_addendum, ephemeral_notice)
            if part
        )
        use_tracking = self._requires_tracked_system_prompt(
            system_prompt_budget_bytes=system_prompt_budget_bytes,
            anchored_doctrine=anchored_doctrine,
        )
        system_prompt = ""
        if not use_tracking:
            system_prompt = self.context_builder.build_system_prompt(
                constitution=constitution,
                include_briefing=include_briefing,
                prompt_adaptation=prompt_adaptation,
                state_of_mind=None,
                system_prompt_addendum=system_prompt_addendum,
            )
            system_prompt = f"{system_prompt}\n\n{ephemeral_notice}"
            # Preserve byte-identical legacy prompt/cache behavior when it fits.
            # If optional bootstrap or briefing text overflows the real
            # tool-aware route ceiling, select the priority formatter before
            # measuring its non-borrowable floor.
            use_tracking = self.counter.count(system_prompt) > system_token_budget

        mandatory_system_tokens = self._measure_mandatory_system_tokens(
            constitution,
            prompt_adaptation,
            anchored_doctrine=anchored_doctrine,
            required_suffix=required_suffix,
            tracked_prompt=use_tracking,
        )
        if mandatory_system_tokens + tools_tokens > total_budget:
            reason = (
                "mandatory governance floor and tool schemas do not fit "
                f"the model payload budget ({mandatory_system_tokens} + "
                f"{tools_tokens} > {total_budget} tokens)"
            )
            assembly = ContextAssembly(
                warnings=[
                    f"DEGRADED MODE: {reason}. The LLM call MUST NOT proceed — "
                    "surface this to the operator."
                ]
            )
            return self._degraded_plan(
                assembly,
                reason=reason,
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
                mode=mode,
            )

        ephemeral_tracking = None
        injected_clauses_for_audit: Optional[List[str]] = None
        dropped_clauses_for_audit: Optional[List[str]] = None
        if use_tracking:
            try:
                ephemeral_tracking = (
                    self.context_builder.build_system_prompt_with_tracking(
                        constitution=constitution,
                        include_briefing=include_briefing,
                        prompt_adaptation=prompt_adaptation,
                        state_of_mind=None,
                        budget_bytes=system_prompt_budget_bytes,
                        budget_tokens=system_token_budget,
                        required_suffix=required_suffix,
                        anchored_doctrine=anchored_doctrine,
                    )
                )
            except MandatorySystemPromptBudgetError as exc:
                reason = str(exc)
                assembly = ContextAssembly(
                    warnings=[
                        f"DEGRADED MODE: {reason}. The LLM call MUST NOT proceed — "
                        "surface this to the operator."
                    ]
                )
                return self._degraded_plan(
                    assembly,
                    reason=reason,
                    mandatory_system_tokens=mandatory_system_tokens,
                    state_of_mind=state_of_mind,
                    mode=mode,
                )
            system_prompt = ephemeral_tracking.prompt
            injected_clauses_for_audit = list(ephemeral_tracking.injected_clauses)
            dropped_clauses_for_audit = list(ephemeral_tracking.dropped_clauses)
            if required_suffix:
                system_prompt = (
                    f"{system_prompt}\n\n{required_suffix}"
                    if system_prompt
                    else required_suffix
                )

        tokens = self.counter.count(system_prompt)

        if tokens + tools_tokens > total_budget:
            reason = (
                "rendered EPHEMERAL system prompt and tool schemas do not fit "
                f"the model payload budget ({tokens} + {tools_tokens} > "
                f"{total_budget} tokens)"
            )
            assembly = ContextAssembly(
                warnings=[
                    f"DEGRADED MODE: {reason}. The LLM call MUST NOT proceed — "
                    "surface this to the operator."
                ]
            )
            return self._degraded_plan(
                assembly,
                reason=reason,
                mandatory_system_tokens=mandatory_system_tokens,
                state_of_mind=state_of_mind,
                mode=mode,
            )

        assembly = ContextAssembly(
            system_prompt=system_prompt,
            warnings=["EPHEMERAL mode: no history available"],
            injected_clauses=injected_clauses_for_audit,
            dropped_clauses=dropped_clauses_for_audit,
        )
        skipped_reason = "excluded by EPHEMERAL privacy mode"
        system_provenance = ["constitution", "bootstrap"]
        if include_briefing:
            system_provenance.append("session_briefing")
        if prompt_adaptation is not None:
            system_provenance.append("prompt_adaptation")
        if system_prompt_addendum:
            system_provenance.append("system_prompt_addendum")
        if anchored_doctrine:
            system_provenance.extend(
                f"anchored_doctrine:{name}" for name in anchored_doctrine
            )
        system_provenance.append("ephemeral_notice")
        sections = {
            "system": ContextSectionPlan(
                name="system",
                destination=SectionDestination.SYSTEM,
                status=SectionStatus.INCLUDED,
                tokens=tokens,
                items=1,
                provenance=tuple(system_provenance),
                details={"subsections": []},
            ),
            "history": ContextSectionPlan(
                name="history",
                destination=SectionDestination.HISTORY,
                status=SectionStatus.SKIPPED,
                tokens=None,
                items=None,
                provenance=("privacy_mode",),
                reason=skipped_reason,
                details={
                    "messages_total": None,
                    "messages_kept_after_pruning": None,
                },
            ),
            "episodes": ContextSectionPlan(
                name="episodes",
                destination=SectionDestination.SYSTEM,
                status=SectionStatus.SKIPPED,
                tokens=None,
                items=None,
                provenance=("privacy_mode",),
                reason=skipped_reason,
                details={"threshold": self.EPISODE_THRESHOLD_MESSAGES},
            ),
            "memories": ContextSectionPlan(
                name="memories",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.SKIPPED,
                tokens=None,
                items=None,
                provenance=("privacy_mode",),
                reason=skipped_reason,
                details={"wired": self.memory_retriever is not None},
            ),
            "rag": ContextSectionPlan(
                name="rag",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.SKIPPED,
                tokens=None,
                items=None,
                provenance=("privacy_mode",),
                reason=skipped_reason,
                details={"chunks": None, "skipped": True},
            ),
            "tools": ContextSectionPlan(
                name="tools",
                destination=SectionDestination.TOOLS,
                status=(
                    SectionStatus.INCLUDED if tools else SectionStatus.EMPTY
                ),
                tokens=tools_tokens,
                items=len(tools or []),
                provenance=("tool_registry", "json_serialized_schemas"),
                details={
                    "estimated": True,
                    "estimation_method": "json-serialized-schemas",
                },
            ),
            "dynamic_context_overhead": ContextSectionPlan(
                name="dynamic_context_overhead",
                destination=SectionDestination.DYNAMIC,
                status=SectionStatus.EMPTY,
                tokens=0,
                items=0,
                provenance=("retrieved_context_wrapper",),
                details={
                    "applies_when": "memories or rag included",
                    "applied": False,
                },
            ),
        }
        return ContextBuildPlan(
            mode=mode,
            model=self.model,
            assembly=assembly,
            sections=sections,
            budget_summary={"mode": "ephemeral"},
            context_limit=context_limit,
            response_reserve=RESPONSE_RESERVE,
            total_budget=total_budget,
            total_tokens=tokens + tools_tokens,
            mandatory_system_tokens=mandatory_system_tokens,
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

    async def compact_session(self, llm_service, preserve_recent: int = 10, force: bool = False, session_id: Optional[str] = None, attribution_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.compact_session(
            llm_service, self.counter, preserve_recent, force, session_id=session_id,
            attribution_session_id=attribution_session_id,
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

    async def summarize_messages(self, llm_service, message_ids: List[int], preserve_key_facts: bool = True, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to ConversationManager."""
        return await self.conversation_manager.summarize_messages(
            llm_service, self.counter, message_ids, preserve_key_facts,
            session_id=session_id,
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

    async def hierarchical_compact(self, llm_service, chunk_size: int = 4000, preserve_recent: int = 5, max_depth: int = 3, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to MemoryManager."""
        return await self.memory_manager.hierarchical_compact(
            llm_service, self.counter, chunk_size, preserve_recent, max_depth,
            session_id=session_id,
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
