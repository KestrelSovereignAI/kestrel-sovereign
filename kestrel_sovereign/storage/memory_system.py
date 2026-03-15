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
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import uuid

from .memory_models import MemoryMetadata, TemporalPattern, MemoryEpisode
from .emotional_tagger import EmotionalTagger
from .temporal_analyzer import TemporalAnalyzer
from .associative_linker import AssociativeLinker
from .memory_retriever import MemoryRetriever
from .memory_consolidator import MemoryConsolidator
from .async_storage import AsyncStorage

logger = logging.getLogger(__name__)


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
        enable_spacy: bool = False
    ):
        """
        Initialize the memory system.

        Args:
            storage: AsyncStorage instance
            agent_id: Agent identifier
            enable_spacy: Enable spaCy for enhanced sentiment analysis
        """
        self.storage = storage
        self.agent_id = agent_id

        # Initialize components
        self.tagger = EmotionalTagger(use_spacy=enable_spacy)
        self.analyzer: Optional[TemporalAnalyzer] = None
        self.linker: Optional[AssociativeLinker] = None
        self.retriever: Optional[MemoryRetriever] = None
        self.consolidator: Optional[MemoryConsolidator] = None

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize memory components that need database access.

        Call this after storage is ready.
        """
        if self._initialized:
            return

        # Components that need database/graph access
        # AsyncStorage uses 'db' not 'database'
        self.analyzer = TemporalAnalyzer(self.storage.db)
        self.linker = AssociativeLinker(self.storage.graph)
        self.retriever = MemoryRetriever(
            self.storage.conversation,
            self.linker
        )
        self.consolidator = MemoryConsolidator(
            self.storage.db,
            self.agent_id,
            graph_store=self.storage.graph,
        )

        self._initialized = True
        logger.info(f"Memory system initialized for agent {self.agent_id}")

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

        # Merge with existing metadata
        enriched = memory_meta.merge_with(existing_metadata or {})

        # Add message ID for concept linking
        enriched["memory_message_id"] = str(uuid.uuid4())

        return enriched

    async def process_message(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full message processing: enrich metadata and extract concepts.

        This is the main entry point for processing new messages.

        Args:
            content: Message content
            role: Message role
            metadata: Existing metadata

        Returns:
            Enriched metadata (concepts extracted as side effect)
        """
        # Enrich with emotional analysis
        enriched = await self.enrich_metadata(content, role, metadata)

        # Extract concepts and link in graph (for user messages)
        if role == "user" and self.linker:
            message_id = enriched.get("memory_message_id", str(uuid.uuid4()))
            concepts = await self.linker.extract_and_link(
                message_id,
                content,
                self.agent_id
            )
            enriched["extracted_concepts"] = concepts

        return enriched

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        emotional_context: Optional[MemoryMetadata] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to query with human-like weighting.

        Args:
            query: Search query
            limit: Max results
            emotional_context: Current emotional state for mood-congruent retrieval

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
            limit=limit
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

    async def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation (call periodically, e.g., nightly).

        This:
        1. Creates narrative episodes from conversation clusters
        2. Detects temporal patterns
        3. Archives fully decayed memories

        Returns:
            Report dict with counts of each operation
        """
        if not self.consolidator:
            logger.warning("Memory consolidator not initialized")
            return {"error": "Consolidator not initialized"}

        return await self.consolidator.run_consolidation()

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
                "memory_consolidator": self.consolidator is not None,
            },
            "scoring_weights": {
                "semantic": MemoryRetriever.WEIGHT_SEMANTIC,
                "emotional": MemoryRetriever.WEIGHT_EMOTIONAL,
                "importance": MemoryRetriever.WEIGHT_IMPORTANCE,
                "recency": MemoryRetriever.WEIGHT_RECENCY,
                "access": MemoryRetriever.WEIGHT_ACCESS,
            },
            "decay_half_life_days": MemoryRetriever.DECAY_HALF_LIFE_DAYS,
        }
