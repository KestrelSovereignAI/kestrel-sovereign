"""
Reflection Feature for agent self-improvement.

Enables the agent to:
1. Reflect on past interactions and identify patterns
2. Generate insights about what worked and what didn't
3. Propose improvements with constitutional approval
4. Apply approved changes to behavior
5. Create GitHub tickets from actionable insights
6. Manage self-model in decentralized storage (Filecoin)

This hooks into the sleep cycle for automatic nightly reflection.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import (
    resolve_feature_conversation_store,
    resolve_feature_database,
)
from kestrel_sovereign.kestrel_config.constants import (
    APPROVAL_TIMEOUT_DEFAULT,
    APPROVAL_TIMEOUT_SHORT,
)

from .models import (
    Insight,
    InsightType,
    ImprovementProposal,
    ChangeType,
    BehaviorRule,
    # Layered reflection models
    ReflectionLayer,
    ReflectionResult,
    LayerResult,
)
from .analyzer import InteractionAnalyzer
from .checks import ArmsChecker, MemoryChecker, MindChecker
from .prioritizer import ActionPrioritizer
from .economics import EconomicGate, ConfigurationError as EconomicsConfigError
from .ticket_creator import TicketCreator, ConfigurationError as TicketConfigError
from .self_model import SelfModelManager, ConfigurationError as SelfModelConfigError
from .db_helpers import ReflectionDatabaseHelper
from .formatters import ReflectionResultFormatter
from .prompts import format_training_summary
from .approval import ApprovalHandler
from .training import TrainingManager
from .ticket_handler import TicketHandler
from .self_model_handler import SelfModelHandler

logger = logging.getLogger(__name__)


class ReflectionFeature(Feature):
    """
    Agent self-reflection and improvement.

    Provides tools for analyzing past interactions, generating insights,
    and proposing self-improvements that require constitutional approval.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Self-reflection and improvement - analyze past interactions, "
            "identify patterns, generate insights, and propose behavioral changes"
        )

    @staticmethod
    def _to_tool_result(
        legacy: Dict[str, Any],
        *,
        ok_confirmation: str,
    ) -> ToolResult:
        """Wrap a legacy success/error dict in a ToolResult envelope.

        Honesty layer 4 (#1042) requires every @tool return to use the
        ToolResult shape. The reflection tools delegate to handlers /
        formatters that still produce rich dicts of the form
        ``{"success": True/False, "error": ..., ...payload...}``. To
        avoid rewriting every handler in this PR, this helper converts
        at the @tool boundary:

          - ``success: False`` (or an ``error`` key without success) →
            ``ToolResult.failed(error)`` carrying the original dict
            under ``data`` so downstream callers can still inspect it.
          - otherwise → ``ToolResult.ok(confirmation, data=dict)``.

        PARTIAL surfaces remain on the methods themselves where the
        domain semantics warrant them (see ``propose_improvement`` for
        rejected-but-stored proposals).
        """
        if not isinstance(legacy, dict):
            return ToolResult.ok(
                confirmation=ok_confirmation,
                data={"raw": legacy},
            )
        if legacy.get("success") is False or (
            "error" in legacy and "success" not in legacy
        ):
            err = legacy.get("error") or "tool reported failure without an error message"
            return ToolResult.failed(err, data=legacy)
        return ToolResult.ok(confirmation=ok_confirmation, data=legacy)

    async def initialize(self):
        """Initialize the reflection feature."""
        self.analyzer: Optional[InteractionAnalyzer] = None
        self._db = None
        self._db_helper: Optional[ReflectionDatabaseHelper] = None
        self._ticket_creator: Optional[TicketCreator] = None
        self._economic_gate: Optional[EconomicGate] = None
        self._self_model_manager: Optional[SelfModelManager] = None
        self._approval_handler: Optional[ApprovalHandler] = None
        self._training_manager: Optional[TrainingManager] = None
        self._ticket_handler: Optional[TicketHandler] = None
        self._self_model_handler: Optional[SelfModelHandler] = None

        # Layered reflection components
        self._arms_checker: Optional[ArmsChecker] = None
        self._memory_checker: Optional[MemoryChecker] = None
        self._mind_checker: Optional[MindChecker] = None
        self._prioritizer = ActionPrioritizer()

        # Initialize database connection and helper
        self._init_database()

        # Initialize core components
        self._init_analyzer()
        self._init_reflection_checkers()
        self._init_optional_services()

        # Wire handlers that compose database + optional services. Has
        # to run AFTER both passes — _init_database used to do this
        # eagerly, but _ticket_creator and _self_model_manager don't
        # exist yet at that point. Result: ``create_improvement_ticket``
        # always returned ``"Ticket handler not available"`` even when
        # GitHub and the database were both healthy.
        self._wire_composite_handlers()

    async def post_all_features_loaded(self, agent):
        """Wire reflection into the sleep cycle after all features are loaded."""
        from kestrel_sovereign.features.reflection.hooks import create_reflection_hook
        agent.reflection_hook = create_reflection_hook(agent)
        if agent.reflection_hook:
            logger.info("Reflection hook enabled for sleep cycle")

    def _init_database(self):
        """Initialize database connection and helper."""
        self._db = resolve_feature_database(self.agent)

        # Initialize database helper
        if self._db:
            agent_id = self._get_agent_id()
            self._db_helper = ReflectionDatabaseHelper(self._db, agent_id)

        # Initialize approval handler
        self._approval_handler = ApprovalHandler(self.agent, self._db_helper)

        # Initialize training manager
        self._training_manager = TrainingManager(self)

        # Composite handlers (ticket, self-model) are wired in
        # _wire_composite_handlers() after _init_optional_services
        # populates _ticket_creator and _self_model_manager.

    def _init_analyzer(self):
        """Initialize the interaction analyzer."""
        conversation_store = (
            resolve_feature_conversation_store(self.agent)
            or getattr(self.agent, 'conversation_store', None)
        )

        # Get episode store (from memory consolidator)
        episode_store = None
        if hasattr(self.agent, 'memory_consolidator'):
            episode_store = self.agent.memory_consolidator

        # Get LLM service
        llm_service = None
        if hasattr(self.agent, 'llm_service'):
            llm_service = self.agent.llm_service

        # Initialize analyzer if we have the required components
        if llm_service and conversation_store:
            agent_id = self._get_agent_id()
            self.analyzer = InteractionAnalyzer(
                llm_service=llm_service,
                conversation_store=conversation_store,
                episode_store=episode_store,
                agent_id=agent_id,
            )
            logger.info("ReflectionFeature initialized with InteractionAnalyzer")
        else:
            logger.warning(
                f"ReflectionFeature initialized without analyzer "
                f"(llm_service={llm_service is not None}, "
                f"conversation_store={conversation_store is not None})"
            )

    def _init_reflection_checkers(self):
        """Initialize layered reflection checkers."""
        self._arms_checker = ArmsChecker(self.agent)
        self._memory_checker = MemoryChecker(self.agent)
        self._mind_checker = MindChecker(self.agent, self.analyzer)
        logger.info("ReflectionFeature initialized with layered checkers")

    def _init_optional_services(self):
        """Initialize optional services (economic gate, ticket creator, self-model manager)."""
        # Initialize economic gate (optional - depends on wallet feature)
        wallet_feature = self._get_feature("wallet")
        if wallet_feature:
            try:
                self._economic_gate = EconomicGate(wallet_feature)
                logger.info("ReflectionFeature: Economic gate initialized")
            except EconomicsConfigError as e:
                logger.warning(f"ReflectionFeature: Economic gate not available: {e}")

        # Initialize ticket creator (optional - depends on GitHub config)
        github_feature = self._get_feature("github")
        github_client = getattr(github_feature, 'client', None) if github_feature else None
        if github_client:
            try:
                self._ticket_creator = TicketCreator(github_client)
                logger.info("ReflectionFeature: Ticket creator initialized")
            except TicketConfigError as e:
                logger.warning(f"ReflectionFeature: Ticket creator not available: {e}")

        # Initialize self-model manager (optional - requires a decentralized storage provider)
        lighthouse_provider = self._get_decentralized_storage_provider()
        agent_did = self._get_agent_id()
        if lighthouse_provider and agent_did:
            try:
                self._self_model_manager = SelfModelManager(
                    storage_provider=lighthouse_provider,
                    agent_did=agent_did,
                    db=self._db,
                )
                logger.info("ReflectionFeature: Self-model manager initialized")
            except SelfModelConfigError as e:
                logger.warning(f"ReflectionFeature: Self-model manager not available: {e}")

    def _wire_composite_handlers(self):
        """Build handlers that need both the database helper AND the
        optional services that come from other features.

        Lives in its own pass because ``_init_database`` and
        ``_init_optional_services`` populate different halves of the
        dependency graph; whichever ran first won't see the other
        half's outputs.
        """
        if self._ticket_creator and self._db_helper:
            self._ticket_handler = TicketHandler(
                self._ticket_creator,
                self._economic_gate,
                self._db_helper,
                self.agent,
            )
            logger.info("ReflectionFeature: TicketHandler wired")
        else:
            missing = []
            if not self._ticket_creator:
                missing.append("ticket_creator (needs github feature + GITHUB_TOKEN)")
            if not self._db_helper:
                missing.append("db_helper (needs reflection database)")
            logger.info(
                "ReflectionFeature: TicketHandler not wired — missing %s",
                ", ".join(missing),
            )

        if self._self_model_manager and self._db_helper:
            self._self_model_handler = SelfModelHandler(
                self._self_model_manager,
                self._economic_gate,
                self._db_helper,
                self.agent,
            )
            logger.info("ReflectionFeature: SelfModelHandler wired")

    def _get_feature(self, name: str):
        """Get a feature from the agent."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature(name)
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get(name)
        return None

    def _get_decentralized_storage_provider(self):
        """
        Return the best available decentralized storage provider.

        Falls back to the agent's storage attribute for forward compatibility.
        """
        lighthouse = (
            getattr(self.agent, 'lighthouse_provider', None)
            or (
                hasattr(self.agent, 'storage')
                and getattr(self.agent.storage, 'lighthouse_provider', None)
            )
        )
        if lighthouse and lighthouse.is_available():
            return lighthouse

        return None

    def _get_agent_id(self) -> str:
        """Get the agent's identifier (DID is the canonical source of truth)."""
        return self.agent.did

    # =========================================================================
    # Core Reflection Tools
    # =========================================================================

    @tool(
        name="reflect",
        description="Perform layered self-reflection: Arms (functional) → Memory (knowledge) → Mind (cognitive) → Actions (priorities)",
        category=ToolCategory.SYSTEM,
        command_prefix="!reflect"
    )
    async def reflect(
        self,
        scope: str = "all",
        depth: str = "normal",
    ) -> ToolResult:
        """
        Perform layered self-reflection.

        Layer 1 (Arms): Physical/functional checks - "Do my components work?"
        Layer 2 (Memory): Knowledge/context checks - "Can I access what I know?"
        Layer 3 (Mind): Cognitive checks - "Is my reasoning producing good outputs?"
        Layer 4 (Action): Prioritized fixes with concrete file:line references

        If a layer has CRITICAL failures, reflection stops and reports immediately.

        Args:
            scope: Time range for Mind layer analysis ('session', 'today', 'week', 'month', 'all')
            depth: Analysis depth for Mind layer ('shallow', 'normal', 'deep')

        Returns:
            ReflectionResult with layer results and prioritized actions
        """
        # Create reflection result
        result = ReflectionResult(
            id=str(uuid.uuid4()),
            trigger="on_demand",
            started_at=datetime.utcnow(),
        )

        try:
            # Layer 1: Arms - Do my components work?
            logger.info("Reflection Layer 1: Arms (functional checks)")
            if self._arms_checker:
                arms_checks = await self._arms_checker.run_all()
                result.arms = LayerResult(
                    layer=ReflectionLayer.ARMS,
                    checks=arms_checks,
                    completed_at=datetime.utcnow(),
                )
                result.layers_completed = 1

                # Stop on critical
                if result.arms.has_critical:
                    logger.warning("Reflection stopped: CRITICAL failure in Arms layer")
                    result.stopped_at_layer = ReflectionLayer.ARMS
                    result.completed_at = datetime.utcnow()
                    result.actions = self._prioritizer.prioritize(result.arms, None, None)
                    return self._reflect_to_tool_result(result)

            # Layer 2: Memory - Can I access what I know?
            logger.info("Reflection Layer 2: Memory (knowledge checks)")
            if self._memory_checker:
                memory_checks = await self._memory_checker.run_all()
                result.memory = LayerResult(
                    layer=ReflectionLayer.MEMORY,
                    checks=memory_checks,
                    completed_at=datetime.utcnow(),
                )
                result.layers_completed = 2

                # Stop on critical
                if result.memory.has_critical:
                    logger.warning("Reflection stopped: CRITICAL failure in Memory layer")
                    result.stopped_at_layer = ReflectionLayer.MEMORY
                    result.completed_at = datetime.utcnow()
                    result.actions = self._prioritizer.prioritize(result.arms, result.memory, None)
                    return self._reflect_to_tool_result(result)

            # Layer 3: Mind - Is my reasoning producing good outputs?
            logger.info(f"Reflection Layer 3: Mind (cognitive checks, depth={depth})")
            if self._mind_checker:
                mind_checks = await self._mind_checker.run_all(depth=depth)
                result.mind = LayerResult(
                    layer=ReflectionLayer.MIND,
                    checks=mind_checks,
                    completed_at=datetime.utcnow(),
                )
                result.layers_completed = 3

            # Layer 4: Action - Prioritize fixes
            logger.info("Reflection Layer 4: Action (prioritization)")
            result.actions = self._prioritizer.prioritize(result.arms, result.memory, result.mind)
            result.completed_at = datetime.utcnow()

            # Persist session and insights
            await self._persist_reflection(result)

            return self._reflect_to_tool_result(result)

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            result.error = str(e)
            result.completed_at = datetime.utcnow()
            await self._persist_reflection(result)
            return self._reflect_to_tool_result(result)

    async def _persist_reflection(self, result: ReflectionResult) -> None:
        """Store reflection session and insights to the database."""
        if not self._db_helper:
            return

        try:
            # Extract insights from Mind layer check details
            insights = []
            if result.mind:
                for check in result.mind.checks:
                    if check.details and "insights" in check.details:
                        for item in check.details["insights"]:
                            try:
                                insights.append(Insight.from_dict(item))
                            except Exception:
                                continue

            # Build and store session
            from .models import ReflectionSession
            session = ReflectionSession(
                id=result.id,
                trigger=result.trigger,
                started_at=result.started_at,
                completed_at=result.completed_at,
                insights=insights,
                error=result.error,
            )
            await self._db_helper.store_session(session)

            # Store individual insights
            for insight in insights:
                await self._db_helper.store_insight(insight, session_id=result.id)

            if insights:
                logger.info(f"Persisted reflection session {result.id} with {len(insights)} insights")

        except Exception as e:
            logger.warning(f"Failed to persist reflection: {e}")

    def _format_result(self, result: ReflectionResult) -> Dict[str, Any]:
        """Format ReflectionResult for API response."""
        return ReflectionResultFormatter.format_reflection_result(result)

    def _reflect_to_tool_result(self, result: ReflectionResult) -> ToolResult:
        """Convert a ReflectionResult into a ToolResult envelope.

        Honesty surfaces:
          - error path → ERROR
          - stopped_at_layer is set (critical failure halted reflection
            before all layers completed) → PARTIAL with the layer name
            in the caveat so the agent must speak "reflection stopped
            early at <layer>" instead of narrating a clean success.
          - critical findings in completed layers → PARTIAL with the
            count so the agent can't narrate "everything's fine" while
            the action list contains CRITICAL items.
        """
        legacy = self._format_result(result)
        summary = legacy.get("summary") or {}
        critical = summary.get("critical_failures", 0) or 0

        if result.error:
            return ToolResult.failed(result.error, data=legacy)

        if result.stopped_at_layer is not None:
            layer_name = getattr(result.stopped_at_layer, "value", str(result.stopped_at_layer))
            return ToolResult.partial(
                confirmation=(
                    f"Reflection ran but stopped early at layer {layer_name!s}"
                ),
                error=(
                    f"a CRITICAL failure in the {layer_name!s} layer halted "
                    "reflection before later layers ran; the action list is "
                    "based on a partial picture. Address the critical failure "
                    "first, then re-run."
                ),
                data=legacy,
            )

        if critical > 0:
            return ToolResult.partial(
                confirmation=(
                    f"Reflection completed (passed={summary.get('total_passed', 0)}, "
                    f"failed={summary.get('total_failed', 0)})"
                ),
                error=(
                    f"{critical} layer(s) reported CRITICAL findings; "
                    "see action list for prioritized fixes"
                ),
                data=legacy,
            )

        return ToolResult.ok(
            confirmation=(
                f"Reflection completed (layers={summary.get('layers_completed', 0)}, "
                f"actions={summary.get('action_count', 0)})"
            ),
            data=legacy,
        )

    @tool(
        name="get_insights",
        description="Get past insights from reflection sessions",
        category=ToolCategory.SYSTEM,
        command_prefix="!insights"
    )
    async def get_insights(
        self,
        type_filter: str = None,
        min_confidence: float = 0.5,
        limit: int = 20,
    ) -> ToolResult:
        """
        Get past insights from reflection sessions.

        Args:
            type_filter: Filter by insight type ('pattern', 'improvement', 'success', 'failure', 'anomaly')
            min_confidence: Minimum confidence threshold (0.0 - 1.0)
            limit: Maximum number of insights to return
        """
        if not self._db_helper:
            return ToolResult.failed("Database not available")

        try:
            insights = await self._db_helper.get_insights(
                type_filter=type_filter,
                min_confidence=min_confidence,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return ToolResult.failed(str(e))

        legacy = ReflectionResultFormatter.format_insights_response(insights)
        count = legacy.get("count", len(insights) if isinstance(insights, list) else 0)
        return ToolResult.ok(
            confirmation=f"Found {count} insight(s)",
            data=legacy,
        )

    @tool(
        name="propose_improvement",
        description="Propose a self-improvement (requires constitutional approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!propose-improvement"
    )
    async def propose_improvement(
        self,
        title: str,
        description: str,
        change_type: str,
        proposed_change: str,
    ) -> ToolResult:
        """
        Propose a self-improvement.

        This requires constitutional approval before being applied.

        Args:
            title: Short title for the improvement
            description: Detailed description of why this improvement is needed
            change_type: Type of change ('prompt', 'behavior', 'tool_usage', 'response_style')
            proposed_change: The specific change to make
        """
        try:
            ct = ChangeType(change_type)
        except ValueError:
            valid = [c.value for c in ChangeType]
            return ToolResult.failed(
                f"Invalid change_type. Valid options: {valid}",
                data={"change_type": change_type},
            )

        proposal = ImprovementProposal(
            id=str(uuid.uuid4()),
            insight_id=None,  # Manual proposal
            title=title,
            description=description,
            change_type=ct,
            proposed_change=proposed_change,
        )

        if self._db_helper:
            await self._db_helper.store_proposal(proposal)

        approved = await self._approval_handler.request_approval(proposal)

        if approved:
            await self._approval_handler.apply_improvement(proposal)
            proposal.applied_at = datetime.utcnow()
            if self._db_helper:
                await self._db_helper.update_proposal(proposal)

        data = {
            "success": True,
            "proposal_id": proposal.id,
            "requires_approval": proposal.requires_approval,
            "approved": proposal.approved,
            "rejection_reason": proposal.rejection_reason,
            "applied": proposal.is_applied,
        }

        # Honesty: a proposal that was REJECTED by constitutional review
        # is recorded but not applied. The legacy code returned
        # ``"success": True`` for the *recording* — but from the agent's
        # standpoint, the improvement did NOT take effect. Surface as
        # PARTIAL so the LLM cannot narrate "self-improvement applied"
        # for a proposal the constitution rejected.
        if proposal.requires_approval and not proposal.approved:
            return ToolResult.partial(
                confirmation=(
                    f"Proposal {proposal.id[:8]} stored (title={title!r})"
                ),
                error=(
                    "constitutional review rejected the proposal; the "
                    "change was not applied. Reason: "
                    f"{proposal.rejection_reason or 'not provided'}"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Proposal {proposal.id[:8]}: "
                + ("approved + applied" if proposal.is_applied else "stored")
            ),
            data=data,
        )

    @tool(
        name="get_behavior_rules",
        description="Get active behavior rules that have been approved",
        category=ToolCategory.SYSTEM,
    )
    async def get_behavior_rules(
        self,
        active_only: bool = True,
    ) -> ToolResult:
        """
        Get behavior rules that modify agent behavior.

        Args:
            active_only: Only return active rules (default: True)
        """
        if not self._db_helper:
            return ToolResult.failed("Database not available")

        try:
            rules = await self._db_helper.get_behavior_rules(active_only=active_only)
        except Exception as e:
            logger.error(f"Failed to get behavior rules: {e}")
            return ToolResult.failed(str(e))

        legacy = ReflectionResultFormatter.format_behavior_rules_response(rules)
        count = legacy.get("count", len(rules) if isinstance(rules, list) else 0)
        return ToolResult.ok(
            confirmation=f"Found {count} behavior rule(s)",
            data=legacy,
        )

    # =========================================================================
    # GitHub Ticket Creation Tools
    # =========================================================================

    @tool(
        name="create_improvement_ticket",
        description="Create a GitHub issue from an actionable insight (requires constitutional approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!create-ticket"
    )
    async def create_improvement_ticket(
        self,
        insight_id: str,
    ) -> ToolResult:
        """
        Create a GitHub issue from an actionable insight.

        This requires:
        - Economic eligibility (paid tier or revenue share)
        - Constitutional approval before creation
        - GITHUB_PAT environment variable configured

        Args:
            insight_id: ID of the insight to create a ticket for
        """
        if not self._ticket_handler:
            return ToolResult.failed(
                "Ticket handler not available - check configuration",
            )

        legacy = await self._ticket_handler.create_improvement_ticket(insight_id)
        url = (legacy or {}).get("issue_url") if isinstance(legacy, dict) else None
        return self._to_tool_result(
            legacy or {},
            ok_confirmation=(
                f"Created improvement ticket"
                + (f" at {url}" if url else "")
                + (f" (insight={insight_id})")
            ),
        )

    # =========================================================================
    # Self-Model Management Tools
    # =========================================================================

    @tool(
        name="get_self_model",
        description="Get the agent's current self-model (personality, communication style, preferences)",
        category=ToolCategory.SYSTEM,
        command_prefix="!self-model"
    )
    async def get_self_model(self) -> ToolResult:
        """
        Get the agent's current self-model.

        The self-model captures:
        - Personality traits (e.g., helpfulness, formality)
        - Communication style preferences
        - Learned user preferences
        - Behavior patterns (successes and failures)
        """
        if not self._self_model_handler:
            return ToolResult.failed(
                "Self-model handler not available - check configuration",
            )

        legacy = await self._self_model_handler.get_self_model()
        return self._to_tool_result(
            legacy or {},
            ok_confirmation="Retrieved self-model",
        )

    @tool(
        name="update_self_model",
        description="Update self-model based on recent insights (requires constitutional approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!update-self-model"
    )
    async def update_self_model(
        self,
        from_session_id: str = None,
    ) -> ToolResult:
        """
        Update the self-model based on recent insights.

        This requires:
        - Economic eligibility (paid tier)
        - Constitutional approval for self-modification
        - LIGHTHOUSE_API_KEY environment variable configured

        Args:
            from_session_id: Optional session ID to get insights from (default: most recent)
        """
        if not self._self_model_handler:
            return ToolResult.failed(
                "Self-model handler not available - check configuration",
            )

        legacy = await self._self_model_handler.update_self_model(from_session_id)
        return self._to_tool_result(
            legacy or {},
            ok_confirmation="Self-model updated",
        )

    # =========================================================================
    # Sleep Integration
    # =========================================================================

    async def on_pre_sleep(self) -> Dict[str, Any]:
        """
        Called before memory consolidation during sleep.

        Performs a shallow reflection on the current session. The
        return shape is the legacy reflection-result dict (not a
        ToolResult); sleep-cycle hooks predate the @tool envelope and
        consume the formatted dict directly.
        """
        logger.info("Running pre-sleep reflection")
        envelope = await self.reflect(scope="session", depth="shallow")
        return envelope.data if envelope.data is not None else {"success": False, "error": envelope.error or ""}

    async def on_post_consolidation(
        self,
        consolidation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Called after memory consolidation during sleep.

        Performs deeper reflection using consolidated episodes. Returns
        the legacy reflection-result dict (see on_pre_sleep note).
        """
        episodes_created = consolidation_result.get("episodes_created", 0)

        if episodes_created > 0:
            logger.info(f"Running post-consolidation reflection ({episodes_created} new episodes)")
            envelope = await self.reflect(scope="today", depth="normal")
            return envelope.data if envelope.data is not None else {"success": False, "error": envelope.error or ""}

        return {"success": True, "skipped": True, "reason": "No new episodes"}

    # =========================================================================
    # Training Cycle (Intensive Improvement)
    # =========================================================================

    @tool(
        name="training_cycle",
        description="Run intensive training cycle: rapid reflection → ticket creation → improvement → verify. Unlike sleep (long-term), this is for active meditation.",
        category=ToolCategory.SYSTEM,
        command_prefix="!train"
    )
    async def training_cycle(
        self,
        iterations: int = 3,
        depth: str = "normal",
        create_tickets: bool = False,
    ) -> ToolResult:
        """
        Run intensive training cycle for rapid self-improvement.

        Args:
            iterations: Number of reflection cycles (default: 3)
            depth: Analysis depth ('quick', 'normal', 'deep')
            create_tickets: Whether to create GitHub issues for action items
        """
        legacy = await self._training_manager.run_training_cycle(
            iterations=iterations,
            depth=depth,
            create_tickets=create_tickets,
        )
        ran = (legacy or {}).get("iterations_completed", iterations) if isinstance(legacy, dict) else iterations
        return self._to_tool_result(
            legacy or {},
            ok_confirmation=f"Training cycle completed ({ran}/{iterations} iterations)",
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================


    async def get_active_guidance(self) -> List[str]:
        """
        Get all active guidance/rules for inclusion in prompts.

        Called by the agent when building prompts to include
        learned behavior modifications.
        """
        if not self._db_helper:
            return []

        return await self._db_helper.get_active_guidance()
