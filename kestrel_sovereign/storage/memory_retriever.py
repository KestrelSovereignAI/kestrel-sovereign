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
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Sequence, Tuple
import json

from .memory_models import MemoryMetadata
from .async_conversation_store import AsyncConversationStore
from .associative_linker import AssociativeLinker
from kestrel_sovereign.security.input_guardrails import extract_raw_user_content

logger = logging.getLogger(__name__)


_ECHO_EDGE_PUNCT = ".,;:!?\"'()[]{}"


def _normalize_for_echo_check(text: str) -> str:
    """Normalize for the near-duplicate echo guard.

    Tokenizes on whitespace, edge-strips sentence-ending punctuation
    and quotes/brackets from each token, collapses to a single space,
    lowercases. Used ONLY to drop a stored row whose content is
    effectively the same string as the current query — NOT for any
    scoring path.

    Edge-strip ONLY sentence punctuation + quotes/brackets, NOT
    operators or modifiers. ``C++`` and ``C`` must stay distinct;
    same for ``version 1.2`` vs ``version 12``. Trying to strip every
    punctuation byte conflates them. The narrower
    ``.,;:!?\"'()[]{}`` set is enough to drop the trivial variants
    the echo guard targets (``color?`` vs ``color``, ``"hello"`` vs
    ``hello``) without touching internal alphanumeric punctuation.
    (Codex P3 on #1481.)
    """
    if not text:
        return ""
    tokens = []
    for tok in text.split():
        tok = tok.strip(_ECHO_EDGE_PUNCT)
        if tok:
            tokens.append(tok.lower())
    return " ".join(tokens)


def _cosine_unit(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cosine similarity normalised into ``[0, 1]``.

    Returns ``None`` when the inputs are unusable (empty, length
    mismatch, zero norm) so the caller can fall back to keyword
    overlap for THIS row without mistaking "no signal" for
    "neutral score." Otherwise returns ``(cos + 1) / 2`` so the
    output sits in the same ``[0, 1]`` band the rest of
    ``_score_semantic`` uses.

    Pure-Python (no numpy) — the retriever rescores ~1000 rows
    per call and a numpy import here adds a few ms of cold-start
    cost per agent boot for no win at this dimension count.
    """
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    raw = dot / math.sqrt(norm_a * norm_b)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


class MemoryRetriever:
    """
    Retrieves memories using human-like weighting.

    Scoring breakdown:
    - semantic: 25% - How relevant is this to the query
    - emotional: 20% - Does the emotional tone match
    - importance: 20% - How important was this marked
    - recency: 15% - How recent, with decay curve
    - certainty: 10% - How certain the claim is (epistemic weight)
    - access: 10% - How often accessed (rehearsal effect)
    """

    # Scoring weights (must sum to 1.0)
    WEIGHT_SEMANTIC = 0.25
    WEIGHT_EMOTIONAL = 0.20
    WEIGHT_IMPORTANCE = 0.20
    WEIGHT_RECENCY = 0.15
    WEIGHT_ACCESS = 0.10
    WEIGHT_CERTAINTY = 0.10

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
        self._access_update_tasks: set[asyncio.Task[None]] = set()

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        emotional_context: Optional[MemoryMetadata] = None,
        limit: int = 10,
        min_score: float = 0.1,
        read_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories with human-like weighting.

        Args:
            query: Search query
            agent_id: Agent ID for scoping
            emotional_context: Current emotional context for matching
            limit: Max results to return
            min_score: Minimum score threshold
            read_only: When True, skip access-count update scheduling so
                callers can estimate the memory block without rehearsal-effect
                writes.

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

        # Compute the query embedding ONCE if the conversation store
        # has an embedding service. Done before the loop so we don't
        # pay an Ollama round-trip per row. Any failure → None, which
        # ``_score_semantic`` reads as "fall back to keyword overlap"
        # for every row in this call.
        query_embedding = await self._embed_query(query)

        # If we have a query embedding, hydrate the row embeddings for
        # the slice in one batched SELECT so we can apply cosine in
        # ``_score_semantic`` without a per-row IO. Rows missing an
        # embedding (legacy, pre-Phase-2-migration, or rows written
        # while Ollama was down) get a None and fall back to keyword
        # overlap naturally.
        row_embeddings: Dict[Any, List[float]] = {}
        if query_embedding is not None:
            ids = [m.get("id") for m in history if m.get("id") is not None]
            # #1477 — only load rows stamped with the current
            # embedding profile id. Cross-profile rows return as
            # absent (no entry in the dict) → ``_score_semantic``
            # naturally falls through to keyword overlap for those
            # rows. ``None`` (no service / no profile metadata)
            # means "no filter" — preserves legacy behavior.
            current_profile_id: Optional[str] = None
            embedding_service_for_profile = getattr(
                self.conversations, "embedding_service", None
            )
            if embedding_service_for_profile is not None and hasattr(
                embedding_service_for_profile, "current_profile_id"
            ):
                try:
                    current_profile_id = (
                        embedding_service_for_profile.current_profile_id()
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "current_profile_id() failed during retriever "
                        "load: %s", exc,
                    )
            try:
                row_embeddings = await self.conversations.get_message_embeddings(
                    ids, embedding_profile_id=current_profile_id,
                )
            except TypeError:
                # Older store version without the keyword (defense
                # against in-process mismatch / tests that stub the
                # store). Fall through to the unfiltered legacy call.
                try:
                    row_embeddings = await self.conversations.get_message_embeddings(
                        ids
                    )
                except Exception as e:
                    logger.warning(
                        "Could not load row embeddings for vector cosine "
                        "(falling back to keyword overlap): %s", e,
                    )
            except Exception as e:
                # Embedding load failure must NOT block retrieval —
                # the keyword path is a complete fallback.
                logger.warning(
                    "Could not load row embeddings for vector cosine "
                    "(falling back to keyword overlap): %s", e,
                )

        # Normalize query for dedup comparison. The echo guard below is
        # the only barrier between a literal prior user question and it
        # resurfacing as a "memory" of itself — so the normalization
        # needs to be aggressive enough to catch trivial variants
        # (different punctuation, extra whitespace, casing) without
        # being so loose it would conflate genuinely different
        # sentences. Strip ASCII punctuation, collapse internal
        # whitespace to single spaces, lowercase. (Codex P2 round 2
        # on #1481 — exact-match was too strict.)
        query_normalized = _normalize_for_echo_check(query)

        # Score each message
        scored: List[Tuple[Dict[str, Any], float]] = []

        for msg in history:
            content = msg.get("content", "")

            # Skip messages that are near-duplicates of the current query.
            # This is the ONLY echo guard now — the prior blanket
            # ``role=user`` skip from #271 was over-broad: it threw out
            # every user-stated biographical fact ("I love sailing on
            # Lake Michigan", "My birthday is April 3rd") along with
            # the questions it was trying to suppress. The injection
            # format in ``agent/memory_manager.py`` now prefixes each
            # memory with ``User:`` / ``Assistant:`` so the LLM reads
            # surfaced user-role content with explicit provenance and
            # won't confuse it with its own prior thoughts. See #1481.
            #
            # User turns are persisted wrapped via ``wrap_user_input``
            # (``<user_input>\n...\n</user_input>``) — strip the wrapper
            # before comparing so the echo guard fires on production
            # rows, not just on synthetic test data. (Codex P2 on PR-1481.)
            comparison_content = content
            if msg.get("role") == "user":
                comparison_content = extract_raw_user_content(content)
            if _normalize_for_echo_check(comparison_content) == query_normalized:
                continue

            score = self._calculate_score(
                content=content,
                query=query,
                metadata=msg.get("metadata", {}),
                emotional_context=emotional_context,
                created_at=msg.get("created_at"),
                expanded_concepts=expanded_concepts,
                query_embedding=query_embedding,
                content_embedding=row_embeddings.get(msg.get("id")),
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

        if not read_only:
            for result in results:
                msg_id = result.get("id")
                if msg_id is not None:
                    self._schedule_access_update(msg_id, agent_id)

        return results

    def _schedule_access_update(self, message_id: int, agent_id: str) -> asyncio.Task[None]:
        """Own rehearsal-effect bookkeeping tasks so shutdown can await them."""
        task = asyncio.create_task(
            self.update_access(message_id, agent_id),
            name=f"memory-access-update-{message_id}",
        )
        self._access_update_tasks.add(task)
        task.add_done_callback(self._access_update_tasks.discard)
        return task

    async def drain_access_updates(self, *, cancel: bool = False) -> None:
        """Wait for scheduled access-count updates before storage lifecycle changes."""
        tasks = set(self._access_update_tasks)
        if not tasks:
            return

        if cancel:
            for task in tasks:
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._access_update_tasks.difference_update(tasks)

    async def shutdown(self) -> None:
        """Cancel pending rehearsal-effect writes during component shutdown."""
        await self.drain_access_updates(cancel=True)

    def _calculate_score(
        self,
        content: str,
        query: str,
        metadata: Dict[str, Any],
        emotional_context: Optional[MemoryMetadata],
        created_at: Optional[str],
        expanded_concepts: List[str],
        query_embedding: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate weighted retrieval score.

        Components:
        - semantic: 25% (keyword + concept overlap, OR vector cosine
          when both query/content embeddings are present)
        - emotional: 20% (mood match)
        - importance: 20% (from metadata)
        - recency: 15% (with decay)
        - certainty: 10% (epistemic weight)
        - access: 10% (rehearsal effect)
        """
        # Parse metadata if string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # 1. Semantic score — vector cosine when we have embeddings
        # for BOTH sides, keyword overlap otherwise. Cosine is a
        # strict upgrade over keyword overlap (semantic similarity
        # without literal token co-occurrence requirement), so a
        # row with an embedding will score higher / lower
        # correctly relative to keyword-only rows in the same call.
        semantic = self._score_semantic(
            content, query, expanded_concepts,
            query_embedding=query_embedding,
            content_embedding=content_embedding,
        )

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

        # 6. Certainty score (epistemic weight)
        certainty = self._score_certainty(metadata)

        # Weighted combination
        total = (
            semantic * self.WEIGHT_SEMANTIC +
            emotional * self.WEIGHT_EMOTIONAL +
            importance * self.WEIGHT_IMPORTANCE +
            recency * self.WEIGHT_RECENCY +
            access * self.WEIGHT_ACCESS +
            certainty * self.WEIGHT_CERTAINTY
        )

        return total

    def _score_semantic(
        self,
        content: str,
        query: str,
        expanded_concepts: List[str],
        query_embedding: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None,
    ) -> float:
        """
        Score semantic relevance.

        Two paths:

        1. **Vector cosine** when both ``query_embedding`` and
           ``content_embedding`` are present (Ollama nomic-embed-text
           or whatever the conversation store is configured for).
           Combined 70% cosine + 30% concept-bonus so the
           concept-expansion path from :class:`AssociativeLinker` still
           rewards related-concept matches the embedding might miss
           on short utterances.
        2. **Keyword overlap** fallback when embeddings aren't
           available — same shape as the original
           ``TODO: can be upgraded to embeddings`` implementation,
           preserved verbatim so a deployment without Ollama keeps
           the prior behaviour.

        Mixing both paths within a single ``retrieve()`` call is
        intentional: rows written before the Phase-2 migration / while
        Ollama was down get keyword scores; new rows get cosine. The
        scores are normalised into the same 0..1 band so this doesn't
        produce a discontinuity in the final ranking.
        """
        if query_embedding is not None and content_embedding is not None:
            cosine = _cosine_unit(query_embedding, content_embedding)
            if cosine is not None:
                content_lower = content.lower()
                concept_score = 0.0
                if expanded_concepts:
                    concept_matches = sum(
                        1 for c in expanded_concepts if c in content_lower
                    )
                    concept_score = min(
                        1.0, concept_matches / len(expanded_concepts)
                    )
                # Same 70/30 split the keyword path uses — keeps the
                # weighting between "core signal" and "concept-expansion
                # bonus" consistent across paths.
                return cosine * 0.7 + concept_score * 0.3

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

    async def _embed_query(self, query: str) -> Optional[List[float]]:
        """Embed the retrieval query with the conversation store's
        configured embedding service.

        Returns ``None`` (and the retriever silently falls back to
        keyword overlap for every row this call) when:

        - The conversation store has no embedding service (opted out
          via ``KESTREL_DISABLE_CONVERSATION_EMBEDDINGS=true``, or
          lazy-acquire failed).
        - The service raises (Ollama outage, model missing, timeout).
        - The query is empty / whitespace.

        Critically the SAME embedding service that WROTE the row
        embeddings must produce the QUERY embedding — different
        models would render cosine meaningless. We pull it from the
        conversation store directly to guarantee that.
        """
        service = getattr(self.conversations, "embedding_service", None)
        if service is None:
            return None
        if not query or not query.strip():
            return None
        try:
            embedding = await service.aembed(query)
        except Exception as e:
            logger.warning(
                "Query embedding failed (falling back to keyword "
                "overlap for this retrieve): %s", e,
            )
            return None
        if not embedding:
            return None
        return list(embedding)

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

    def _score_certainty(self, metadata: Dict[str, Any]) -> float:
        """Score based on epistemic certainty.

        Higher-certainty claims score higher — the agent is more
        confident in them and they should surface more readily.
        Messages without epistemic metadata get a neutral 0.5.
        """
        claim_certainty = metadata.get("claim_certainty")
        if claim_certainty is None:
            return 0.5
        return float(claim_certainty)

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

        Uses the store's atomic JSON-set increment under the hood so
        two concurrent retrievals of the same memory both register
        instead of racing on a read-modify-write of the same counter.

        Args:
            message_id: Database ID of the message
            agent_id: Agent ID for verification (currently unused; store is
                      already agent-scoped)
        """
        if not self.conversations or message_id is None:
            return

        try:
            await self.conversations.atomic_increment_metadata_counter(
                message_id,
                counter_field="access_count",
                timestamp_field="last_accessed",
            )
        except Exception as e:
            # Never let rehearsal-effect bookkeeping break retrieval
            logger.warning(f"update_access failed for message {message_id}: {e}")

    async def update_applied(
        self,
        message_id: int,
        agent_id: str,
    ) -> None:
        """
        Record that a retrieved memory was demonstrably applied.

        Distinct from ``update_access``: access is incremented when the
        retriever scores a memory into the context window, applied is
        incremented when the agent's downstream loop (typically the
        reflection / pre-sleep hook) attests that the memory was
        load-bearing for the decision that followed.  Without this
        distinction, decoration that's retrieved every session is
        indistinguishable from memory that's actually steering the
        agent.

        Auto-detection of "applied" is intentionally out of scope of
        this primitive; this method is the write path and reflection
        decides when to call it.  See #1326.

        Uses the same atomic increment helper ``update_access`` uses so
        concurrent reflection hooks marking the same memory as applied
        can't lose increments to a read-modify-write race.

        Args:
            message_id: Database ID of the message
            agent_id: Agent ID for verification (currently unused; store
                is already agent-scoped, parameter matches
                ``update_access`` shape so the two write paths feel
                symmetric to callers).
        """
        if not self.conversations or message_id is None:
            return

        try:
            await self.conversations.atomic_increment_metadata_counter(
                message_id,
                counter_field="applied_count",
                timestamp_field="last_applied",
            )
        except Exception as e:
            # Same safety as update_access — bookkeeping must not
            # break the calling loop.
            logger.warning(f"update_applied failed for message {message_id}: {e}")


def calculate_decay(
    created_at: str,
    importance: float = 0.5,
    access_count: int = 0,
    decay_protected: bool = False,
    half_life_days: int = 30,
    applied_count: int = 0,
) -> float:
    """
    Calculate current memory strength based on decay.

    Standalone function for use in consolidation and other contexts.

    The ``access_count`` and ``applied_count`` parameters carry distinct
    signal:

    * ``access_count`` — the memory was retrieved into the agent's
      context window.  This is the rehearsal-effect signal from
      Ebbinghaus and is correctly modest in magnitude — every load
      shouldn't materially extend a memory's lifespan.
    * ``applied_count`` — the memory demonstrably changed what the
      agent did next (populated via ``MemorySystem.mark_applied`` from
      reflection / pre-sleep hooks; see #1326).  A higher boost
      coefficient than ``access_count`` rewards load-bearing memories
      over ones that merely keep getting recalled.

    Concretely the boost curves are:

    * access:  ``1.0 + log10(n + 1) * 0.5``  (n=1 → +0.15, n=10 → +0.52, n=100 → +1.0)
    * applied: ``1.0 + log10(n + 1) * 1.0``  (n=1 → +0.30, n=10 → +1.00, n=100 → +2.00)

    The boosts multiply, so a memory that's been both accessed and
    applied compounds — the system rewards being *useful* over being
    *familiar*, but doesn't punish familiarity either.

    Args:
        created_at: ISO timestamp of memory creation
        importance: Importance score (0.0 to 1.0)
        access_count: Number of times retrieved
        decay_protected: If True, returns 1.0 (no decay)
        half_life_days: Base half-life in days
        applied_count: Number of times the memory was demonstrably
            applied to a downstream decision.  Defaults to 0 so existing
            callers that pre-date #1326 get unchanged behavior.

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

        # Access boosts half-life (rehearsal effect — modest)
        if access_count > 0:
            import math
            access_boost = 1.0 + math.log10(access_count + 1) * 0.5
            effective_half_life *= access_boost

        # Applied boosts half-life more strongly than access — being
        # load-bearing is a stronger signal than being retrieved.
        if applied_count > 0:
            import math
            applied_boost = 1.0 + math.log10(applied_count + 1) * 1.0
            effective_half_life *= applied_boost

        # Exponential decay
        decay = 0.5 ** (days_old / effective_half_life)

        return decay

    except (ValueError, TypeError):
        return 0.5
