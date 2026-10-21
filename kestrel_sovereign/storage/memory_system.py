"""
Human-Like Memory System - Unified Interface.

Orchestrates all memory components:
- EmotionalTagger: Sentiment and importance analysis
- TemporalAnalyzer: Pattern detection
- AssociativeLinker: Concept graph building
- MemoryRetriever: Weighted retrieval
- MemoryConsolidator: Nightly maintenance

This facade provides a simple API for KestrelAgent integration.

Usage:
    memory = MemorySystem(storage, agent_id)
    await memory.initialize()

    # On message save:
    enriched_meta = await memory.enrich_metadata(content, role)

    # On retrieval:
    memories = await memory.retrieve(query, emotional_context)

    # Background maintenance:
    report = await memory.consolidate()
"""
import asyncio
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Any, List, Optional

from .memory_models import MemoryMetadata, TemporalPattern, MemoryEpisode
from .emotional_tagger import EmotionalTagger
from .temporal_analyzer import TemporalAnalyzer
from .associative_linker import AssociativeLinker
from .memory_retriever import MemoryRetriever
from kestrel_sovereign.llm.decisions.config import parse_decision_selector
from kestrel_sovereign.llm.provider_policy import config_provenance
from kestrel_sovereign.execution_custody import is_execution_control_error

from .memory_answerability import (
    AnswerabilityGate,
    DecisionAnswerabilityGate,
    LLMAnswerabilityGate,
)
from .memory_consolidator import MemoryConsolidator
from .consolidation_drain import (
    ISOLATION_CALLER_CANCELLED,
    ISOLATION_CLEANUP_DEADLINE,
    ISOLATION_CONNECTION_CLOSED,
    ISOLATION_WAIT_FAILED,
    ConsolidationDrain,
    reconcile_consolidation_drains,
)
from .db.sqlite import (
    CANCELLATION_CLEANUP_FAILED,
    CANCELLATION_CLEANUP_FINISHED,
    CANCELLATION_CLEANUP_ISOLATED,
    CANCELLATION_CLEANUP_RETIRED,
    CANCELLATION_CLEANUP_RETIRING,
)
from .schema_router import SchemaRouter
from .async_storage import AsyncStorage
from kestrel_sovereign.config import load_section

logger = logging.getLogger(__name__)


# Consolidation walks the active memory corpus and may backfill provider
# embeddings, so allow substantially more time than an ordinary recall while
# still bounding the single maintenance chokepoint.
DEFAULT_CONSOLIDATION_TIMEOUT_SECONDS = 30 * 60.0


#: What became of the database work a timed-out pass left running (#2978).
ABANDONED_WORK_NONE = "none"
ABANDONED_WORK_FINISHED = CANCELLATION_CLEANUP_FINISHED
ABANDONED_WORK_FAILED = CANCELLATION_CLEANUP_FAILED
ABANDONED_WORK_ISOLATED = CANCELLATION_CLEANUP_ISOLATED
ABANDONED_WORK_RETIRED = CANCELLATION_CLEANUP_RETIRED
ABANDONED_WORK_RETIRING = CANCELLATION_CLEANUP_RETIRING

_ABANDONED_WORK_OUTCOMES = {
    None: "the pass was abandoned and in-flight database work may still be "
    "draining",
    ABANDONED_WORK_NONE: "the pass was abandoned with no database work left "
    "running",
    ABANDONED_WORK_FINISHED: "the pass was abandoned, and the database work "
    "it left running finished before the memory lock was released",
    ABANDONED_WORK_FAILED: "the pass was abandoned, and rolling back the "
    "database work it left running failed, so the database connection "
    "refuses all work until it is reconnected",
    ABANDONED_WORK_ISOLATED: "the pass was abandoned while its database work "
    "was still running, so the database connection refuses all work until "
    "that work finishes",
    ABANDONED_WORK_RETIRED: "the pass was abandoned, and the database "
    "connection was closed under the work it left running",
    ABANDONED_WORK_RETIRING: "the pass was abandoned, and the database "
    "connection was closed under the work it left running, which keeps "
    "running until that connection's worker thread exits",
}


class MemoryConsolidationTimeoutError(TimeoutError):
    """The configured consolidation deadline expired.

    Distinct from provider/socket ``TimeoutError`` exceptions raised by work
    inside consolidation. Callers may translate this deadline breach without
    misreporting an unrelated I/O timeout as the global maintenance bound.

    ``abandoned_work`` says what became of database work the pass left
    running (one of the ``ABANDONED_WORK_*`` values, ``None`` when unknown),
    and ``drain_record`` names the durable record of it, if one was written.
    """

    def __init__(
        self,
        timeout_seconds: float,
        abandoned_work: Optional[str] = None,
        drain_record: Optional[str] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.abandoned_work = abandoned_work
        self.drain_record = drain_record
        super().__init__(
            "Memory consolidation exceeded its configured deadline of "
            f"{timeout_seconds:g} seconds; "
            + _ABANDONED_WORK_OUTCOMES[abandoned_work]
        )


class MemoryApplicationNotRecorded(RuntimeError):
    """``mark_applied`` wrote nothing (#3544).

    There was no store to write to, or no row of this agent with the id. A
    caller that took the call as done would count an attestation that never
    reached the memory's decay protection.
    """


def _positive_timeout_seconds(
    config: Dict[str, Any], key: str, default: float
) -> float:
    """Validate a positive finite timeout from the retrieval config."""
    timeout = config.get(key, default)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError(f"retrieval.{key} must be positive")
    return float(timeout)


def _consolidation_timeout_seconds() -> float:
    """Load and validate the global memory-consolidation deadline."""
    config = load_section("retrieval") or {}
    return _positive_timeout_seconds(
        config,
        "memory_consolidation_timeout_seconds",
        DEFAULT_CONSOLIDATION_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True)
class AnswerabilitySettings:
    """``[retrieval] memory_answerability_*``.

    ``backend`` picks the judge: ``"chat"`` (``LLMService.generate``, model
    ``model`` in the chat selector grammar) or ``"decision"``
    (``LLMService.decide``, model ``decision_model`` in the decisions grammar,
    #3424). Each model key belongs to one backend only.
    """

    enabled: bool
    timeout_seconds: float
    model: Optional[str]
    backend: str = "chat"
    decision_model: Optional[str] = None


ANSWERABILITY_BACKENDS = ("chat", "decision")
DECISION_BACKENDS = ANSWERABILITY_BACKENDS


def _decision_backend_setting(
    config: Dict[str, Any],
    *,
    backend_key: str,
    decision_model_key: str,
    chat_model_key: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Validate one ``<x>_backend`` / ``<x>_decision_model`` pair in ``[retrieval]``.

    Shared by every retrieval caller that can run on ``chat`` or ``decision``
    (#3424). A decision model only means something to the decision backend,
    and a chat model only to the chat backend: setting the other backend's key
    is an error naming the right key, never a value read under the wrong
    selector grammar.
    """
    backend = config.get(backend_key, "chat")
    decision_model = config.get(decision_model_key)
    if backend not in DECISION_BACKENDS:
        raise ValueError(f'retrieval.{backend_key} must be "chat" or "decision"')
    if decision_model is not None:
        if not isinstance(decision_model, str) or not decision_model.strip():
            raise ValueError(f"retrieval.{decision_model_key} must be a selector string")
        try:
            parse_decision_selector(decision_model)
        except ValueError as error:
            raise ValueError(f"retrieval.{decision_model_key}: {error}") from error
    if backend == "chat" and decision_model is not None:
        raise ValueError(
            f"retrieval.{decision_model_key} applies only when "
            f'{backend_key} = "decision"'
        )
    if backend == "decision" and chat_model_key and config.get(chat_model_key) is not None:
        raise ValueError(
            f"retrieval.{chat_model_key} is the chat backend's model; "
            f"with the decision backend use {decision_model_key} "
            "(<vendor>[:<route>][/<model>])"
        )
    return backend, decision_model.strip() if decision_model else None


def _answerability_settings() -> AnswerabilitySettings:
    """Load and validate the global retrieval answerability controls."""
    config = load_section("retrieval") or {}
    enabled = config.get("memory_answerability_gate", True)
    timeout = config.get("memory_answerability_timeout_seconds", 12.0)
    model = config.get("memory_answerability_model")
    if not isinstance(enabled, bool):
        raise ValueError("retrieval.memory_answerability_gate must be boolean")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or float(timeout) <= 0
    ):
        raise ValueError(
            "retrieval.memory_answerability_timeout_seconds must be positive"
        )
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("retrieval.memory_answerability_model must be a model string")
    backend, decision_model = _decision_backend_setting(
        config,
        backend_key="memory_answerability_backend",
        decision_model_key="memory_answerability_decision_model",
        chat_model_key="memory_answerability_model",
    )
    return AnswerabilitySettings(
        enabled=enabled,
        timeout_seconds=float(timeout),
        model=model.strip() if model else None,
        backend=backend,
        decision_model=decision_model,
    )


@dataclass(frozen=True)
class AttestationSettings:
    """``[retrieval] memory_attestation_*`` for the pre-sleep application check."""

    backend: str = "chat"
    decision_model: Optional[str] = None


def _attestation_settings() -> AttestationSettings:
    """Load and validate the sleep-time memory attestation backend (#3495)."""
    config = load_section("retrieval") or {}
    backend, decision_model = _decision_backend_setting(
        config,
        backend_key="memory_attestation_backend",
        decision_model_key="memory_attestation_decision_model",
    )
    return AttestationSettings(backend=backend, decision_model=decision_model)


def _build_answerability_gate(
    llm_service: Any, settings: AnswerabilitySettings
) -> AnswerabilityGate:
    # The model is the operator's ``[retrieval]`` setting, not the agent's
    # choice (#2652).
    if settings.backend == "decision":
        return DecisionAnswerabilityGate(
            llm_service,
            timeout_seconds=settings.timeout_seconds,
            model_override=settings.decision_model,
            model_override_writer=config_provenance(
                via="[retrieval] memory_answerability_decision_model"
            ),
        )
    return LLMAnswerabilityGate(
        llm_service,
        timeout_seconds=settings.timeout_seconds,
        model_override=settings.model,
        model_override_writer=config_provenance(
            via="[retrieval] memory_answerability_model"
        ),
    )


def _routing_suppressed(metadata: Optional[Dict[str, Any]]) -> bool:
    """Callers (e.g. MemoryFeature in EPHEMERAL privacy mode) can opt out
    of schema routing by setting `skip_schema_routing=True` in metadata.
    The primary gate is privacy_wrapper — this flag is a belt-and-braces
    fallback for callers that know routing should not persist structure."""
    if not metadata:
        return False
    return bool(metadata.get("skip_schema_routing"))


class MemorySystem:
    """
    Human-like memory system facade.

    Integrates emotional tagging, temporal patterns, concept graphs,
    and weighted retrieval into a single interface.
    """

    def __init__(
        self,
        storage: AsyncStorage,
        agent_id: str,
        enable_spacy: bool = False,
        privacy_storage=None,
        transition_lock=None,
    ):
        """
        Initialize the memory system.

        Args:
            storage: AsyncStorage instance used for the memory system's direct
                SQL (temporal analysis, decay, conversation reads).
            agent_id: Agent identifier
            enable_spacy: Enable spaCy for enhanced sentiment analysis
            privacy_storage: Optional privacy-enforcing storage facade. When
                provided, durable graph writes (episode nodes, concept links,
                schema-routed action/decision nodes) flow through its
                privacy-governing ``.graph`` proxy, and the consolidator gates
                its direct ``memory_episodes`` write on the same policy — so
                manual / scheduled consolidation cannot leak user-derived memory
                into durable storage in a volatile privacy mode (#2672). ``None``
                (raw storage, tests) leaves writes ungoverned, as before.
            transition_lock: The agent's privacy-transition mutex, forwarded to
                the consolidator so episode repair holds it across its
                mode check and its durable writes (#2856). ``None`` runs
                unguarded.
        """
        self.storage = storage
        self.agent_id = agent_id
        self._privacy_storage = privacy_storage
        # Components are built in initialize(), so this has to survive __init__.
        self._transition_lock = transition_lock

        # Initialize components
        self.tagger = EmotionalTagger(use_spacy=enable_spacy)
        self.analyzer: Optional[TemporalAnalyzer] = None
        self.linker: Optional[AssociativeLinker] = None
        self.retriever: Optional[MemoryRetriever] = None
        self.consolidator: Optional[MemoryConsolidator] = None
        self.router: Optional[SchemaRouter] = None

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize memory components that need database access.

        Call this after storage is ready.
        """
        if self._initialized:
            return

        self._reconcile_consolidation_drain()

        # Components that need database/graph access
        # AsyncStorage uses 'db' not 'database'.
        #
        # Durable graph writes route through the privacy-governing proxy when a
        # privacy facade was injected, so volatile-mode writes (episode nodes,
        # concept links, schema-routed nodes) fail closed at the storage boundary
        # (#2672). Direct SQL still uses the raw storage handed in as ``storage``
        # so temporal/decay reads stay unchanged and warning-free.
        governed_graph = (
            self._privacy_storage.graph
            if self._privacy_storage is not None
            else self.storage.graph
        )
        self.analyzer = TemporalAnalyzer(self.storage.db)
        self.linker = AssociativeLinker(governed_graph)
        llm_service = getattr(self.storage, "llm_service", None)
        answerability = _answerability_settings()
        self.retriever = MemoryRetriever(
            self.storage.conversation,
            self.linker,
            answerability_gate=(
                _build_answerability_gate(llm_service, answerability)
                if llm_service is not None and answerability.enabled
                else None
            ),
            answerability_enabled=answerability.enabled,
        )
        self.consolidator = MemoryConsolidator(
            self.storage.db,
            self.agent_id,
            graph_store=governed_graph,
            # Reuse the agent-scoped LLM service for episode embeddings +
            # semantic recall (#1674 P2). None-safe: recall degrades to
            # keyword search when no embedding provider is configured.
            llm_service=llm_service,
            # Gate the direct memory_episodes write on the privacy policy so
            # manual / scheduled consolidation can't persist user-derived
            # episodes in a volatile mode (#2672).
            persist_policy=self._privacy_storage,
            # #2850: the consolidator reads conversation_history with its own
            # SQL, so it needs the store that owns decryption — without it the
            # at-rest envelope was tokenized straight into episode topics.
            conversation_store=self.storage.conversation,
            # #2856: repair checks the privacy mode then awaits decryption and
            # several writes; the lock keeps a transition from landing in that
            # gap.
            transition_lock=self._transition_lock,
        )

        # Schema-aware routing: promote extracted structure (action items,
        # decisions, person-interaction enrichment) after concept linking.
        # Everything typed lives in the graph now, so table creation is a
        # no-op — the call is kept for forward-compat if a later router
        # variant needs setup.
        self.router = SchemaRouter(
            graph=governed_graph,
            db=self.storage.db,
            agent_id=self.agent_id,
        )
        try:
            await self.router.ensure_tables()
        except Exception as e:
            logger.warning("SchemaRouter init failed: %s", e)
            self.router = None

        self._initialized = True
        logger.info(f"Memory system initialized for agent {self.agent_id}")

    def _reconcile_consolidation_drain(self) -> None:
        """Report drain records no pass will settle on its own (#2978).

        A process that dies while a timed-out pass's database work drains
        never marks that pass's record done. Startup reads every record for
        the database and logs each one that does not say its work settled,
        with who holds it. It changes none of them: the only write it can
        cause is a pass of this very process retrying its own settlement.
        """
        report = reconcile_consolidation_drains(getattr(self.storage, "db", None))
        if report is None:
            return
        for status in report.unsettled:
            logger.warning(
                "Memory consolidation drain record found at startup of agent "
                "%s: %s",
                self.agent_id,
                status.describe(),
            )
        for unreadable in report.unreadable:
            logger.warning(
                "At startup of agent %s: %s",
                self.agent_id,
                unreadable.describe(),
            )

    async def shutdown(self) -> None:
        """Shut down owned memory background work before storage closes."""
        if self.retriever:
            await self.retriever.shutdown()

    async def tag_message(
        self,
        message_id: int,
        content: str,
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Tag a single stored message with emotional metadata (Phase 1 - inline).

        This is CPU-bound keyword matching via EmotionalTagger, safe to call
        synchronously in the request path. It writes the enriched metadata
        directly to the message row so it is available for future retrieval.

        Args:
            message_id: Database row ID of the message to tag
            content: Message text (already stored; passed to avoid re-read)
            role: Message role ('user' or 'assistant')

        Returns:
            The enriched metadata dict that was written
        """
        enriched = await self.enrich_metadata(content, role)

        # Write tags back to the stored message
        conv_store = self.storage.conversation
        if conv_store:
            try:
                await conv_store.update_message_metadata(message_id, enriched)
            except Exception as e:
                logger.error(
                    f"Failed to tag message {message_id}: {e}", exc_info=True
                )

        return enriched

    async def enrich_metadata(
        self,
        content: str,
        role: str = "user",
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich message metadata with emotional and importance analysis.

        Call this when adding a conversation message to get enhanced metadata.

        Args:
            content: Message content
            role: Message role ('user' or 'assistant')
            existing_metadata: Any existing metadata to merge with

        Returns:
            Enriched metadata dict ready for storage
        """
        # Analyze emotional content
        memory_meta = await self.tagger.analyze(content, role)

        # Message identity belongs to the conversation store.  Do not mint a
        # second UUID here: graph nodes, schema artifacts, and metadata must all
        # refer to the persisted conversation_history.id.
        return memory_meta.merge_with(existing_metadata or {})

    async def process_message(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        message_id: int | str,
    ) -> Dict[str, Any]:
        """
        Full message processing: enrich metadata and extract concepts.

        This is the main entry point for processing new messages.

        Args:
            content: Message content
            role: Message role
            metadata: Existing metadata
            message_id: Canonical persisted conversation-history row ID.

        Returns:
            Enriched metadata (concepts extracted as side effect)
        """
        # Enrich with emotional analysis
        enriched = await self.enrich_metadata(content, role, metadata)

        derived = await self.link_and_route_message(
            message_id=message_id,
            content=content,
            role=role,
            metadata=enriched,
        )
        enriched.update(derived)

        return enriched

    async def link_and_route_message(
        self,
        *,
        message_id: int | str,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create graph-derived memory state for one persisted message.

        This is the canonical post-storage graph path.  The caller supplies the
        real conversation row ID so message nodes, action items, decisions, and
        interaction edges all share one identity.
        """
        if role != "user" or not self.linker:
            return {}
        if message_id is None or str(message_id).strip() == "":
            raise ValueError("message_id is required for memory graph processing")

        canonical_id = str(message_id)
        linked_concepts = await self.linker.extract_and_link(
            canonical_id,
            content,
            self.agent_id,
        )
        derived: Dict[str, Any] = {
            "extracted_concepts": [concept.label for concept in linked_concepts]
        }

        if self.router and not _routing_suppressed(metadata):
            try:
                derived["schema_routing"] = await self.router.route(
                    message_id=canonical_id,
                    content=content,
                    concepts=linked_concepts,
                    role=role,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning("Schema routing failed for message %s: %s", canonical_id, exc)

        return derived

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        emotional_context: Optional[MemoryMetadata] = None,
        min_relevance: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to query with human-like weighting.

        Args:
            query: Search query
            limit: Max results
            emotional_context: Current emotional state for mood-congruent retrieval
            min_relevance: Semantic/lexical/associative eligibility floor. Set
                to 0 only for intentionally relevance-free emotional recall.

        Returns:
            List of message dicts with retrieval_score
        """
        if not self.retriever:
            logger.warning("Memory retriever not initialized")
            return []

        return await self.retriever.retrieve(
            query,
            self.agent_id,
            emotional_context=emotional_context,
            limit=limit,
            min_relevance=min_relevance,
        )

    async def retrieve_important(
        self,
        min_importance: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve highly important memories.

        Args:
            min_importance: Minimum importance threshold
            limit: Max results

        Returns:
            List of important messages
        """
        if not self.retriever:
            return []

        return await self.retriever.retrieve_important(
            self.agent_id,
            min_importance=min_importance,
            limit=limit
        )

    async def retrieve_by_emotion(
        self,
        emotion: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories tagged with specific emotion.

        Args:
            emotion: Emotion category (joy, sadness, etc.)
            limit: Max results

        Returns:
            List of messages with that emotion
        """
        if not self.retriever:
            return []

        return await self.retriever.retrieve_by_emotion(
            self.agent_id,
            emotion=emotion,
            limit=limit
        )

    async def get_temporal_context(self) -> Dict[str, Any]:
        """
        Get current temporal context and matching patterns.

        Returns info like "User is most active late at night" if
        current time matches a detected pattern.

        Returns:
            Dict with current time context and relevant patterns
        """
        if not self.analyzer:
            return {
                "current_time_of_day": TemporalAnalyzer.get_time_of_day(),
                "current_day_of_week": TemporalAnalyzer.get_day_of_week(),
                "matching_patterns": [],
                "pattern_insights": [],
            }

        return await self.analyzer.get_current_context(self.agent_id)

    async def get_associated_concepts(
        self,
        concept: str,
        min_strength: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get concepts associated with a given concept.

        Args:
            concept: The concept to find associations for
            min_strength: Minimum association strength

        Returns:
            List of associated concepts with strengths
        """
        if not self.linker:
            return []

        return await self.linker.get_associated_concepts(
            concept,
            self.agent_id,
            min_strength=min_strength
        )

    async def get_concept_network(
        self,
        concept: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get network of concepts around a central concept.

        Useful for visualization and understanding how concepts connect.

        Args:
            concept: Central concept
            depth: How many hops to explore

        Returns:
            Dict with 'nodes' and 'edges' for visualization
        """
        if not self.linker:
            return {"center": concept, "nodes": [], "edges": []}

        return await self.linker.get_concept_network(
            concept,
            self.agent_id,
            depth=depth
        )

    @asynccontextmanager
    async def durable_writes(self) -> AsyncIterator[bool]:
        """Hold the privacy-transition lock; say whether a durable write may run.

        The same permission and lock the consolidator's durable writes use
        (#2672, #2856). A writer that checks the mode and then awaits its
        write does both inside this block, so a transition to EPHEMERAL /
        ISOLATED cannot land between them. Yields ``False`` in a mode that
        forbids persistence: the block must then write nothing.
        """
        from .privacy_wrapper import optional_transition_lock, persistent_writes_allowed

        async with optional_transition_lock(self._transition_lock):
            yield persistent_writes_allowed(self._privacy_storage)

    async def mark_applied(
        self,
        message_id: int,
        *,
        reason: Optional[str] = None,
    ) -> None:
        """
        Record that a previously retrieved memory was demonstrably applied.

        Distinct from the implicit retrieval bookkeeping done in
        ``retrieve()``: that increments ``access_count`` whenever a
        memory scores into the context window.  ``mark_applied`` is the
        explicit attestation that the memory was load-bearing for the
        agent's next decision, populated by reflection / pre-sleep
        hooks (or any other surface that can witness application
        directly).  See #1326.

        Args:
            message_id: Database ID of the message whose memory was applied.
            reason: Optional human-readable rationale for the
                attestation (e.g. ``"steered tool choice"``,
                ``"matched recall constraint"``).  Captured for audit /
                future telemetry; not yet persisted into metadata —
                reflection-side aggregation is a follow-up.  Logged
                here so the application context is visible in agent
                traces even pre-persistence.

        Returns only once the increment is written (#3544): committed, or,
        inside a caller's transaction, part of it. A caller that records the
        attestation anywhere else (the sleep hook's marks) may rely on that.

        Raises:
            MemoryApplicationNotRecorded: Nothing was written: no retriever
                or store, or no row of this agent with ``message_id``.
            Exception: The store's write failed; raised as the store raised it.
        """
        if not self.retriever:
            raise MemoryApplicationNotRecorded(
                "memory retriever not initialized; nothing to mark applied"
            )
        if not await self.retriever.update_applied(message_id, self.agent_id):
            raise MemoryApplicationNotRecorded(
                f"no memory {message_id} of agent {self.agent_id} to mark applied"
            )
        if reason:
            logger.info(
                "memory.mark_applied agent=%s message_id=%s reason=%s",
                self.agent_id, message_id, reason,
            )

    async def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation (call periodically, e.g., nightly).

        This:
        1. Creates narrative episodes from conversation clusters
        2. Detects temporal patterns
        3. Archives fully decayed memories
        4. Forgets (deletes) episodes decayed past the delete threshold (#1674)

        This is the SINGLE consolidation chokepoint — both the nightly ``sleep``
        cycle and the manual ``!memory consolidate`` tool route through here, so
        forgetting lives in one place rather than scattered across per-memory
        crons. The forgetting deletion tier is opt-in/off and best-effort.

        Returns:
            Report dict with counts of each operation

        Raises:
            MemoryConsolidationTimeoutError: The configured consolidation and
                forgetting deadline expired. Cancellation abandons the
                coroutine, but an aiosqlite statement already running on its
                worker thread keeps running; this returns only once that work
                has finished or the connection is isolated from later work
                (see :meth:`_settle_abandoned_database_work`).
        """
        if not self.consolidator:
            logger.warning("Memory consolidator not initialized")
            return {"error": "Consolidator not initialized"}

        timeout_seconds = _consolidation_timeout_seconds()
        deadline: Optional[asyncio.Timeout] = None
        try:
            # Keep the deadline at this single chokepoint so the nightly sleep
            # cycle and every tool caller inherit identical bounds. This
            # timeout cancels the current task; it does not create a child task
            # that could outlive task-owned ResourceLock.MEMORY.
            async with asyncio.timeout(timeout_seconds) as deadline:
                report = dict(await self.consolidator.run_consolidation() or {})
                # Only ride a SUCCESSFUL pass: run_consolidation reports many
                # internal failures as {"error": ...} instead of raising, and
                # destructive forgetting must never follow a failed/partial
                # consolidation. A privacy-skipped pass (volatile mode, #2672)
                # likewise short-circuits the forgetting tier — the whole
                # durable path is gated, not just the writes.
                report["episodes_deleted"] = (
                    0
                    if ("error" in report or report.get("skipped"))
                    else await self._forget_decayed_episodes()
                )
                return report
        except TimeoutError as exc:
            # Only the configured deadline: a provider or socket TimeoutError
            # raised inside the pass leaves the deadline unexpired.
            if deadline is None or not deadline.expired():
                raise
            logger.error(
                "Memory consolidation timed out after %gs; the pass was "
                "abandoned",
                timeout_seconds,
            )
            # Still inside the caller's MEMORY lock: settle what the cancelled
            # pass left running before the caller can release it.
            abandoned_work, drain_record = (
                await self._settle_abandoned_database_work(timeout_seconds)
            )
            raise MemoryConsolidationTimeoutError(
                timeout_seconds,
                abandoned_work,
                None if drain_record is None else str(drain_record),
            ) from exc

    async def _settle_abandoned_database_work(
        self, timeout_seconds: float
    ) -> tuple[str, Optional[Path]]:
        """Settle the database work a timed-out pass left running (#2978).

        Cancelling an aiosqlite await does not stop a statement already
        running on the connection's worker thread. Releasing MEMORY at once
        would let a later pass queue its reads and writes on that same
        connection behind work nobody owns any more. So this keeps the lock
        until the backend's retained rollback proves the abandoned statement
        has returned, waiting no longer than that rollback's own deadline.
        When the wait runs out, fails, or the caller cancels it, the
        connection is isolated first -- every later operation fails closed
        until the work finishes -- and only then is the lock released.

        When the connection is closed under the work before that rollback
        runs, the work ends only with the connection's worker thread: the
        record stays isolated until that thread has exited, however long
        after the close that is.

        This pass's own durable record, beside the database file, tracks the
        work until it settles: the database itself refuses work meanwhile, so
        it cannot hold the evidence. Work already pending on the connection
        when the deadline fired is waited for and recorded the same way,
        whoever abandoned it: the pass cannot tell it apart from its own.

        Returns what became of the work (an ``ABANDONED_WORK_*`` value) and
        the record's path, or ``None`` when no record could be written.
        """
        db = getattr(self.storage, "db", None)
        if db is None or getattr(
            db, "cancellation_cleanup_pending", False
        ) is not True:
            return ABANDONED_WORK_NONE, None

        record = ConsolidationDrain(
            db, agent_id=self.agent_id, deadline_seconds=timeout_seconds
        )
        # Registered while the work is still pending, so its settlement is
        # recorded however the wait below ends, even after the lock is gone.
        if not db.add_cancellation_cleanup_callback(record.settle):
            return ABANDONED_WORK_NONE, None
        record.write()
        logger.warning(
            "Memory consolidation timed out after %gs with database work "
            "still running; holding the memory lock until it finishes "
            "(record: %s)",
            timeout_seconds,
            record.path,
        )
        try:
            cleanup = await db.await_cancellation_cleanup()
        except BaseException as exc:
            # The caller's cancellation (or a failed wait) still propagates
            # unchanged; it just cannot leave the abandoned work sharing the
            # connection.
            db.quarantine_cancellation_cleanup()
            record.isolate(
                ISOLATION_CALLER_CANCELLED
                if isinstance(exc, asyncio.CancelledError)
                else ISOLATION_WAIT_FAILED
            )
            raise
        if cleanup == CANCELLATION_CLEANUP_ISOLATED:
            record.isolate(ISOLATION_CLEANUP_DEADLINE)
        elif cleanup == CANCELLATION_CLEANUP_RETIRING:
            # Settled by the backend once the worker thread exits.
            record.isolate(ISOLATION_CONNECTION_CLOSED)
        else:
            # The backend's settlement callback normally ran first; this is
            # a no-op then.
            record.settle(cleanup)
        return cleanup, None if record.persisted_state is None else record.path

    async def _forget_decayed_episodes(self) -> int:
        """Deletion tier of the forgetting curve (#1674).

        Deletes episodes whose importance- and access-scaled decay strength has
        fallen below ``[forgetting].delete_threshold`` and that are past
        ``[forgetting].delete_grace_days``. Opt-in: a no-op unless
        ``[forgetting].enabled``. Best-effort — a failure here must not fail the
        consolidation it rides on (episodes are retried next pass). Returns the
        number of episodes forgotten.
        """
        from kestrel_sovereign.storage.retention import load_forgetting_config

        config = load_forgetting_config()
        if not config["enabled"]:
            return 0
        storage = self.storage
        if storage is None or not hasattr(storage, "purge_decayed_episodes"):
            return 0
        try:
            return await storage.purge_decayed_episodes(
                delete_threshold=config["delete_threshold"],
                grace_days=config["grace_days"],
                reason="forgetting",
            )
        except Exception as e:  # noqa: BLE001 - forgetting must not fail consolidation
            if is_execution_control_error(e):
                raise
            logger.warning("forgetting: episode deletion tier failed: %s", e)
            return 0

    async def get_episodes(
        self,
        limit: int = 10
    ) -> List[MemoryEpisode]:
        """
        Get stored narrative episodes.

        Args:
            limit: Max episodes to return

        Returns:
            List of MemoryEpisode objects
        """
        if not self.consolidator:
            return []

        return await self.consolidator.get_episodes(limit=limit)

    async def get_patterns(
        self,
        pattern_type: Optional[str] = None
    ) -> List[TemporalPattern]:
        """
        Get detected temporal patterns.

        Args:
            pattern_type: Optional filter (time_preference, day_preference, etc.)

        Returns:
            List of TemporalPattern objects
        """
        if not self.analyzer:
            return []

        return await self.analyzer.get_patterns(
            self.agent_id,
            pattern_type=pattern_type
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the memory system state.

        Returns:
            Dict with component status and basic stats
        """
        return {
            "agent_id": self.agent_id,
            "initialized": self._initialized,
            "components": {
                "emotional_tagger": True,
                "temporal_analyzer": self.analyzer is not None,
                "associative_linker": self.linker is not None,
                "memory_retriever": self.retriever is not None,
                "answerability_gate": bool(
                    self.retriever and self.retriever.answerability_enabled
                ),
                "memory_consolidator": self.consolidator is not None,
            },
            "scoring_weights": {
                "semantic": MemoryRetriever.WEIGHT_SEMANTIC,
                "emotional": MemoryRetriever.WEIGHT_EMOTIONAL,
                "importance": MemoryRetriever.WEIGHT_IMPORTANCE,
                "recency": MemoryRetriever.WEIGHT_RECENCY,
                "certainty": MemoryRetriever.WEIGHT_CERTAINTY,
                "access": MemoryRetriever.WEIGHT_ACCESS,
            },
            "decay_half_life_days": MemoryRetriever.DECAY_HALF_LIFE_DAYS,
            "answerability": (
                dict(self.retriever.answerability_stats) if self.retriever else {}
            ),
        }
