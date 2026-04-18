"""
Human-like memory retrieval with weighted scoring.

Retrieves memories using human-like weighting:
- Semantic relevance (30%)
- Emotional congruence (25%)
- Importance (20%)
- Recency with decay (15%)
- Access frequency (10%)

This creates retrieval that feels like human memory:
emotionally charged, important events stick better.

WHEN TO USE THIS vs AsyncRAGStore
---------------------------------
Use MemoryRetriever (this module) when:
  - Searching CONVERSATION HISTORY and message-level memories
  - You want emotional weighting, importance, and Ebbinghaus decay applied
  - Content is experiential/episodic (what was said, felt, remembered)
  - Examples: "recall what we discussed about X", "find emotionally important moments",
    "what does the user typically feel about Y"

Use AsyncRAGStore (storage/async_rag_store.py) when:
  - Searching INDEXED DOCUMENTS (uploaded files, ingested knowledge bases)
  - You need vector similarity search over chunks of static content
  - Content is referential/factual (not conversational)
  - Examples: "find sections of the user guide about X", "search uploaded PDFs"

The two systems intentionally do NOT share an interface — they answer
different questions about different data. RAG = "what does the document say?"
Memory = "what did we experience together?"

See docs/architecture/MEMORY_SYSTEM.md for the full decision matrix.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

from .memory_models import MemoryMetadata
from .async_conversation_store import AsyncConversationStore
from .associative_linker import AssociativeLinker

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves memories using human-like weighting.

    Scoring breakdown:
    - semantic: 30% - How relevant is this to the query
    - emotional: 25% - Does the emotional tone match
    - importance: 20% - How important was this marked
    - recency: 15% - How recent, with decay curve
    - access: 10% - How often accessed (rehearsal effect)
    """

    # Scoring weights (must sum to 1.0)
    WEIGHT_SEMANTIC = 0.30
    WEIGHT_EMOTIONAL = 0.25
    WEIGHT_IMPORTANCE = 0.20
    WEIGHT_RECENCY = 0.15
    WEIGHT_ACCESS = 0.10

    # Decay parameters (Ebbinghaus-inspired)
    DECAY_HALF_LIFE_DAYS = 30  # Memory strength halves every 30 days

    def __init__(
        self,
        conversation_store: AsyncConversationStore,
        linker: Optional[AssociativeLinker] = None
    ):
        """
        Initialize retriever.

        Args:
            conversation_store: For accessing conversation history
            linker: Optional AssociativeLinker for concept expansion
        """
        self.conversations = conversation_store
        self.linker = linker

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        emotional_context: Optional[MemoryMetadata] = None,
        limit: int = 10,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories with human-like weighting.

        Args:
            query: Search query
            agent_id: Agent ID for scoping
            emotional_context: Current emotional context for matching
            limit: Max results to return
            min_score: Minimum score threshold

        Returns:
            List of message dicts with 'score' field added,
            sorted by score descending
        """
        # Get conversation history
        # Note: AsyncConversationStore's get_conversation_history doesn't take agent_id
        # because it's already scoped via self.agent_id in the store
        history = await self.conversations.get_conversation_history(limit=1000)

        if not history:
            return []

        # Get expanded concepts if linker available
        expanded_concepts: List[str] = []
        if self.linker:
            expanded_concepts = await self.linker.find_concepts_for_query(
                query, agent_id
            )

        # Normalize query for dedup comparison
        query_normalized = query.strip().lower()

        # Score each message
        scored: List[Tuple[Dict[str, Any], float]] = []

        for msg in history:
            # Skip user messages — they're questions/requests, not knowledge.
            # Only assistant and system messages contain useful recall content.
            if msg.get("role") == "user":
                continue

            content = msg.get("content", "")

            # Skip messages that are near-duplicates of the current query
            # (prevents echoing back the user's own question from a prior turn)
            if content.strip().lower() == query_normalized:
                continue

            score = self._calculate_score(
                content=content,
                query=query,
                metadata=msg.get("metadata", {}),
                emotional_context=emotional_context,
                created_at=msg.get("created_at"),
                expanded_concepts=expanded_concepts,
            )

            if score >= min_score:
                scored.append((msg, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top results with scores
        results = []
        for msg, score in scored[:limit]:
            result = dict(msg)
            result["retrieval_score"] = round(score, 4)
            results.append(result)

        # Rehearsal effect: bump access_count on the messages we actually
        # surfaced. Fire-and-forget so it never blocks retrieval.
        for result in results:
            msg_id = result.get("id")
            if msg_id is not None:
                asyncio.create_task(self.update_access(msg_id, agent_id))

        return results

    def _calculate_score(
        self,
        content: str,
        query: str,
        metadata: Dict[str, Any],
        emotional_context: Optional[MemoryMetadata],
        created_at: Optional[str],
        expanded_concepts: List[str],
    ) -> float:
        """
        Calculate weighted retrieval score.

        Components:
        - semantic: 30% (keyword + concept overlap)
        - emotional: 25% (mood match)
        - importance: 20% (from metadata)
        - recency: 15% (with decay)
        - access: 10% (rehearsal effect)
        """
        # Parse metadata if string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # 1. Semantic score (keyword overlap + concept match)
        semantic = self._score_semantic(content, query, expanded_concepts)

        # 2. Emotional score (mood congruence)
        emotional = self._score_emotional(metadata, emotional_context)

        # 3. Importance score (from metadata)
        importance = metadata.get("importance", 0.5)

        # Pinned memory boost -- agent-pinned memories always score high
        if metadata.get("decay_protected"):
            importance = max(importance, 0.9)

        # 4. Recency score (with decay)
        recency = self._score_recency(created_at, importance)

        # 5. Access score (rehearsal effect)
        access = self._score_access(metadata)

        # Weighted combination
        total = (
            semantic * self.WEIGHT_SEMANTIC +
            emotional * self.WEIGHT_EMOTIONAL +
            importance * self.WEIGHT_IMPORTANCE +
            recency * self.WEIGHT_RECENCY +
            access * self.WEIGHT_ACCESS
        )

        return total

    def _score_semantic(
        self,
        content: str,
        query: str,
        expanded_concepts: List[str]
    ) -> float:
        """
        Score semantic relevance.

        Uses keyword overlap for now, can be upgraded to embeddings.
        """
        content_lower = content.lower()
        query_lower = query.lower()

        # Split into words
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        # Remove very common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you",
                      "to", "and", "of", "in", "it", "that", "this", "for"}
        query_words -= stop_words
        content_words -= stop_words

        if not query_words:
            return 0.5  # Neutral if no meaningful query words

        # Keyword overlap
        overlap = len(query_words & content_words)
        keyword_score = min(1.0, overlap / len(query_words))

        # Concept match bonus
        concept_score = 0.0
        if expanded_concepts:
            concept_matches = sum(
                1 for c in expanded_concepts if c in content_lower
            )
            concept_score = min(1.0, concept_matches / len(expanded_concepts))

        # Combine: 70% keyword, 30% concept
        return keyword_score * 0.7 + concept_score * 0.3

    def _score_emotional(
        self,
        metadata: Dict[str, Any],
        emotional_context: Optional[MemoryMetadata]
    ) -> float:
        """
        Score emotional congruence.

        Memories matching current emotional state are retrieved easier
        (mood-congruent recall).
        """
        if not emotional_context:
            return 0.5  # Neutral if no context

        memory_valence = metadata.get("emotional_valence", 0.0)
        context_valence = emotional_context.emotional_valence

        # Same-direction valence is a match
        # Both positive or both negative
        if memory_valence * context_valence > 0:
            # Stronger match for stronger emotions
            match_strength = min(abs(memory_valence), abs(context_valence))
            return 0.5 + match_strength * 0.5
        elif memory_valence * context_valence < 0:
            # Opposite valence - lower score
            return 0.3
        else:
            # One or both neutral
            return 0.5

    def _score_recency(
        self,
        created_at: Optional[str],
        importance: float
    ) -> float:
        """
        Score recency with Ebbinghaus-inspired decay.

        Important memories decay slower (higher importance = longer half-life).
        """
        if not created_at:
            return 0.5  # Neutral if no timestamp

        try:
            # Parse timestamp
            if isinstance(created_at, str):
                # Handle various ISO formats
                created_at = created_at.replace("Z", "+00:00")
                if "+" not in created_at and "-" not in created_at[10:]:
                    created = datetime.fromisoformat(created_at)
                    created = created.replace(tzinfo=timezone.utc)
                else:
                    created = datetime.fromisoformat(created_at)
            else:
                created = created_at

            now = datetime.now(timezone.utc)
            days_old = (now - created).total_seconds() / 86400

            # Importance extends half-life: 1.0 to 3.0x multiplier
            half_life = self.DECAY_HALF_LIFE_DAYS * (1.0 + importance * 2.0)

            # Exponential decay: strength = 0.5 ^ (days / half_life)
            decay = 0.5 ** (days_old / half_life)

            return decay

        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse timestamp {created_at}: {e}")
            return 0.5

    def _score_access(self, metadata: Dict[str, Any]) -> float:
        """
        Score based on access frequency (rehearsal effect).

        Frequently accessed memories are easier to retrieve.
        """
        access_count = metadata.get("access_count", 0)

        # Logarithmic scaling (diminishing returns)
        # 0 accesses = 0.0, 10 accesses = ~0.77, 100 accesses = ~1.0
        import math
        if access_count <= 0:
            return 0.0
        return min(1.0, math.log10(access_count + 1) / 2)

    async def retrieve_by_emotion(
        self,
        agent_id: str,
        emotion: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories tagged with specific emotion.

        Args:
            agent_id: Agent ID
            emotion: Emotion category (e.g., "joy", "sadness")
            limit: Max results

        Returns:
            List of matching messages sorted by importance
        """
        history = await self.conversations.get_conversation_history(limit=1000)

        matching = []
        for msg in history:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            categories = metadata.get("emotional_categories", [])
            if emotion in categories:
                msg_copy = dict(msg)
                msg_copy["importance"] = metadata.get("importance", 0.5)
                matching.append(msg_copy)

        # Sort by importance descending
        matching.sort(key=lambda x: x.get("importance", 0.5), reverse=True)

        return matching[:limit]

    async def retrieve_important(
        self,
        agent_id: str,
        min_importance: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve highly important memories.

        Args:
            agent_id: Agent ID
            min_importance: Minimum importance threshold
            limit: Max results

        Returns:
            List of important messages sorted by importance
        """
        history = await self.conversations.get_conversation_history(limit=1000)

        important = []
        for msg in history:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            importance = metadata.get("importance", 0.5)
            if importance >= min_importance:
                msg_copy = dict(msg)
                msg_copy["importance"] = importance
                msg_copy["importance_reasons"] = metadata.get("importance_reasons", [])
                important.append(msg_copy)

        # Sort by importance descending
        important.sort(key=lambda x: x.get("importance", 0.5), reverse=True)

        return important[:limit]

    async def update_access(
        self,
        message_id: int,
        agent_id: str
    ) -> None:
        """
        Update access count for a retrieved message.

        Called when a message is retrieved to strengthen the memory.
        This implements the rehearsal effect — accessed memories
        decay slower (see calculate_decay below).

        Args:
            message_id: Database ID of the message
            agent_id: Agent ID for verification (currently unused; store is
                      already agent-scoped)
        """
        if not self.conversations or message_id is None:
            return

        try:
            # Read current access_count atomically via the conversation store
            row = await self.conversations.db.fetchone(
                "SELECT metadata FROM conversation_history WHERE id = ? AND agent_id = ?",
                (message_id, self.conversations.agent_id),
            )
            if not row:
                return

            current_meta = json.loads(row[0]) if row[0] else {}
            new_count = (current_meta.get("access_count") or 0) + 1

            await self.conversations.update_message_metadata(
                message_id,
                {
                    "access_count": new_count,
                    "last_accessed": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            # Never let rehearsal-effect bookkeeping break retrieval
            logger.warning(f"update_access failed for message {message_id}: {e}")


def calculate_decay(
    created_at: str,
    importance: float = 0.5,
    access_count: int = 0,
    decay_protected: bool = False,
    half_life_days: int = 30
) -> float:
    """
    Calculate current memory strength based on decay.

    Standalone function for use in consolidation and other contexts.

    Args:
        created_at: ISO timestamp of memory creation
        importance: Importance score (0.0 to 1.0)
        access_count: Number of times accessed
        decay_protected: If True, returns 1.0 (no decay)
        half_life_days: Base half-life in days

    Returns:
        Memory strength from 0.0 to 1.0
    """
    if decay_protected:
        return 1.0

    try:
        # Parse timestamp
        if isinstance(created_at, str):
            created_at = created_at.replace("Z", "+00:00")
            if "+" not in created_at and "-" not in created_at[10:]:
                created = datetime.fromisoformat(created_at)
                created = created.replace(tzinfo=timezone.utc)
            else:
                created = datetime.fromisoformat(created_at)
        else:
            created = created_at

        now = datetime.now(timezone.utc)
        days_old = (now - created).total_seconds() / 86400

        # Importance extends half-life
        effective_half_life = half_life_days * (1.0 + importance * 2.0)

        # Access boosts half-life (rehearsal effect)
        if access_count > 0:
            import math
            access_boost = 1.0 + math.log10(access_count + 1) * 0.5
            effective_half_life *= access_boost

        # Exponential decay
        decay = 0.5 ** (days_old / effective_half_life)

        return decay

    except (ValueError, TypeError):
        return 0.5
