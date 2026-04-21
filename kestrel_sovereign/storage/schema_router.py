"""Schema-aware routing: promote extracted structure to typed storage.

Inspired by OB1's schema-aware routing pattern. Runs after the existing
emotional/importance/temporal tagging and concept linking, and routes
extracted structure to typed graph nodes:

- `action_item` nodes — state-machine entities with status / assignee /
  due_date properties. Everything typed lives in the graph for
  consistency; node_type is indexed so `get_nodes_by_type` is fast.
- `decision` nodes — mirrors the skill pattern from #643.
- Enriched `mentions` edge properties — sentiment and topics on the
  existing message→concept edge. No new edge label: the existing
  mentions edge IS the interaction record.

Person resolution uses a 3-pass matcher (exact → fuzzy first-name →
collision detection) and flags ambiguous matches as `status=pending`
for human confirmation rather than auto-merging.

Per #628, runs only when privacy mode allows — EPHEMERAL and ISOLATED
skip routing entirely because the underlying storage shouldn't exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .async_graph_store import AsyncGraphStore, GraphNode

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

ACTION_ITEM_NODE_TYPE = "action_item"
DECISION_NODE_TYPE = "decision"

ACTION_ITEM_STATUSES = ("pending", "done", "cancelled")

# Extractors are regex/keyword based for the first cut. An LLM-powered
# extractor can replace these later without changing the routing surface.

ACTION_ITEM_PATTERNS = [
    # First-person commitments
    r"\bI (?:need to|have to|must|should|will|'ll|am going to|'m going to|plan to|want to) ([^.!?\n]+)",
    r"\bI(?:'ve)? got to ([^.!?\n]+)",
    # Explicit TODO markers
    r"\bTODO:? ([^.!?\n]+)",
    r"\b(?:remind me to|don'?t forget to) ([^.!?\n]+)",
    # Promises
    r"\bI'?ll ([^.!?\n]+)",
]

DECISION_PATTERNS = [
    # Anchor on an explicit decision verb or noun to keep precision up.
    # Plain future tense ("I'm going to X") is captured by the action-item
    # extractor; promoting it to a decision here would double-classify
    # most commitments and dilute decision recall.
    r"\bI(?:'ve)? decided (?:to |that |on )?([^.!?\n]+)",
    r"\bwe(?:'ve)? decided (?:to |that |on )?([^.!?\n]+)",
    r"\bmy decision (?:is |was )([^.!?\n]+)",
    r"\bgoing with ([^.!?\n]+)",
    r"\bsticking with ([^.!?\n]+)",
    r"\bcommitting to ([^.!?\n]+)",
    r"\bwe'?re going with ([^.!?\n]+)",
]

# Naive positive/negative sentiment cues for interaction enrichment.
# This intentionally does NOT try to reproduce the full emotional tagger —
# it's a cheap per-mention signal, not a document-level sentiment score.
POSITIVE_CUES = frozenset([
    "love", "like", "grateful", "thanks", "appreciate", "happy", "glad",
    "proud", "enjoyed", "wonderful", "great", "awesome",
])
NEGATIVE_CUES = frozenset([
    "hate", "frustrated", "angry", "annoyed", "disappointed", "sad",
    "worried", "upset", "hurt", "betrayed", "stressed",
])


# =============================================================================
# Data models
# =============================================================================


@dataclass
class ActionItem:
    """An extracted action item."""
    id: str
    agent_id: str
    source_message_id: Optional[str]
    text: str
    status: str  # "pending" | "done" | "cancelled"
    assignee_concept_id: Optional[str]
    due_date: Optional[str]
    confidence: float
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_message_id": self.source_message_id,
            "text": self.text,
            "status": self.status,
            "assignee_concept_id": self.assignee_concept_id,
            "due_date": self.due_date,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


@dataclass
class PersonMatch:
    """Result of person resolution."""
    concept_id: Optional[str]
    status: str  # "new" | "exact" | "fuzzy" | "pending"
    candidates: List[str]  # populated when status == "pending"


# =============================================================================
# Person resolver (3-pass)
# =============================================================================


class PersonResolver:
    """Three-pass person concept resolution.

    Pass 1 — exact match on normalized name.
    Pass 2 — fuzzy first-name match. "Rob" → existing "Robert" IF there is
             exactly one person concept whose first token starts with or
             equals the input's first token.
    Pass 3 — collision detection. If multiple candidates share a first name,
             do NOT auto-link. Return status="pending" with the ambiguous
             candidate list for human confirmation. Never silently merge.
    """

    def __init__(self, graph: AsyncGraphStore):
        self.graph = graph

    async def resolve(self, name: str, agent_id: str) -> PersonMatch:
        """Resolve a mentioned person name to a concept_id or a pending flag."""
        normalized = _normalize_person_name(name)
        if not normalized:
            return PersonMatch(concept_id=None, status="new", candidates=[])

        # Pull all existing person concepts for this agent.
        existing = await self._list_person_concepts(agent_id)

        # Pass 1 — exact
        exact_id = next(
            (cid for cid, label in existing if _normalize_person_name(label) == normalized),
            None,
        )
        if exact_id:
            return PersonMatch(concept_id=exact_id, status="exact", candidates=[])

        # Pass 2/3 — first-name fuzzy + collision
        first = normalized.split()[0] if normalized else ""
        if not first:
            return PersonMatch(concept_id=None, status="new", candidates=[])

        first_matches = [
            (cid, label)
            for cid, label in existing
            if _first_name_matches(first, label)
        ]
        if len(first_matches) == 1:
            return PersonMatch(
                concept_id=first_matches[0][0], status="fuzzy", candidates=[]
            )
        if len(first_matches) > 1:
            return PersonMatch(
                concept_id=None,
                status="pending",
                candidates=[cid for cid, _ in first_matches],
            )

        return PersonMatch(concept_id=None, status="new", candidates=[])

    async def _list_person_concepts(self, agent_id: str) -> List[Tuple[str, str]]:
        """Return list of (concept_id, label) for person concepts of this agent."""
        rows = await self.graph.db.fetchall(
            """
            SELECT node_id, label FROM graph_nodes
            WHERE node_type = 'concept' AND node_id LIKE ?
            """,
            (f"concept:{agent_id}:%",),
        )
        return [(row[0], row[1]) for row in rows]


def _normalize_person_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    cleaned = re.sub(r"[^\w\s]", "", name or "").strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _first_name_matches(first: str, label: str) -> bool:
    """True if the first token of `label` matches `first` with enough evidence.

    To avoid false merges like "Al" → "Alice" we require at least 3
    characters of shared prefix, regardless of which side is shorter.
    """
    candidate_first = _normalize_person_name(label).split()
    if not candidate_first:
        return False
    cand = candidate_first[0]
    min_shared = 3
    if len(first) < min_shared or len(cand) < min_shared:
        # Too little evidence to fuzzy-link.
        return cand == first
    if cand == first:
        return True
    if cand.startswith(first) and len(first) >= min_shared:
        return True
    if first.startswith(cand) and len(cand) >= min_shared:
        return True
    return False


# =============================================================================
# Extractors
# =============================================================================


class ActionItemExtractor:
    """Regex-based action item extraction.

    Intentionally simple — captures first-person commitments, TODOs, and
    reminders. An LLM-powered extractor can replace this later without
    changing the surrounding routing code.
    """

    def extract(self, content: str) -> List[str]:
        items: List[str] = []
        seen: set[str] = set()
        for pattern in ACTION_ITEM_PATTERNS:
            for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                text = match.group(len(match.groups())).strip().strip(",;")
                if not text or len(text) < 3:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(text)
        return items


class DecisionExtractor:
    """Regex-based decision extraction."""

    def extract(self, content: str) -> List[str]:
        decisions: List[str] = []
        seen: set[str] = set()
        for pattern in DECISION_PATTERNS:
            for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                text = match.group(len(match.groups())).strip().strip(",;")
                if not text or len(text) < 3:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                decisions.append(text)
        return decisions


def extract_interaction_sentiment(content: str) -> Tuple[Optional[str], List[str]]:
    """Return (sentiment, topics) for an interaction enrichment.

    Sentiment is one of: "positive", "negative", "mixed", None.
    Topics is a short list of keyword nouns surfaced from the message.
    """
    lower = content.lower()
    pos = sum(1 for cue in POSITIVE_CUES if re.search(rf"\b{cue}\b", lower))
    neg = sum(1 for cue in NEGATIVE_CUES if re.search(rf"\b{cue}\b", lower))

    if pos and neg:
        sentiment = "mixed"
    elif pos:
        sentiment = "positive"
    elif neg:
        sentiment = "negative"
    else:
        sentiment = None

    # Topics: simple noun-ish keyword extraction — lowercase words 4+ chars
    # that aren't stopwords and not already captured as sentiment cues.
    topic_candidates = re.findall(r"\b[a-z]{4,}\b", lower)
    stop = _TOPIC_STOPWORDS | POSITIVE_CUES | NEGATIVE_CUES
    topics: List[str] = []
    seen: set[str] = set()
    for word in topic_candidates:
        if word in stop or word in seen:
            continue
        seen.add(word)
        topics.append(word)
        if len(topics) >= 5:
            break
    return sentiment, topics


_TOPIC_STOPWORDS = frozenset([
    "about", "after", "again", "along", "also", "always", "around", "because",
    "been", "before", "being", "between", "both", "could", "does", "doing",
    "during", "each", "even", "ever", "every", "from", "have", "here", "just",
    "know", "like", "many", "more", "most", "much", "must", "never", "often",
    "only", "over", "really", "said", "should", "since", "some", "still",
    "such", "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "under", "until", "very", "what", "when",
    "where", "which", "while", "will", "with", "would", "your", "yours",
    "would", "going", "gonna", "wanted", "think", "thought",
])


# =============================================================================
# Main router
# =============================================================================


class SchemaRouter:
    """Promote extracted structure to typed graph nodes.

    Runs after concept linking in the message pipeline. Reads the message
    content plus the concept list the linker produced, and writes:

    - `action_item` graph nodes with status/due_date/assignee properties
    - `decision` graph nodes
    - enriched properties on the existing `mentions` edges

    Never creates new concept nodes — operates on what the linker built.
    All artifacts share one storage model (graph_nodes + graph_edges)
    so the architecture has one typed-entity pattern, not two.
    """

    def __init__(
        self,
        graph: AsyncGraphStore,
        db,
        agent_id: str,
    ):
        self.graph = graph
        self.db = db
        self.agent_id = agent_id
        self.action_extractor = ActionItemExtractor()
        self.decision_extractor = DecisionExtractor()
        self.person_resolver = PersonResolver(graph)

    async def ensure_tables(self) -> None:
        """No-op.

        Action items, decisions, and interactions all live in the graph
        (graph_nodes and graph_edges). The core schema in async_database.py
        creates those tables plus the node_type/label indexes the router
        depends on for fast typed-entity lookup. This method exists so
        MemorySystem.initialize can call it unconditionally in case a
        future router variant needs setup.
        """
        return None

    async def route(
        self,
        message_id: str,
        content: str,
        concepts: List[str],
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route extracted structure from a message.

        Returns a summary dict for inclusion in the message's enriched
        metadata. Best-effort: a failure in one lane doesn't block the
        others — the routing pass is advisory enrichment, not critical
        path for storing the message.

        Args:
            message_id: Unique id of the source message.
            content: Raw message text.
            concepts: Concept labels the linker extracted.
            role: Message role (only "user" messages are routed).
            metadata: Optional enriched metadata dict (from EmotionalTagger).
                If present, ``claim_source``, ``claim_certainty``, and
                ``temporal_validity`` are propagated to decision and
                action_item nodes so epistemic provenance flows from the
                message to the claim node.
        """
        if role != "user":
            return {"action_items": 0, "decisions": 0, "interactions": 0}

        # Extract epistemic fields from message metadata for claim nodes.
        epistemic: Dict[str, Any] = {}
        if metadata:
            for key in ("claim_source", "claim_certainty", "temporal_validity"):
                val = metadata.get(key)
                if val is not None:
                    epistemic[key] = val

        summary: Dict[str, Any] = {
            "action_items": 0,
            "decisions": 0,
            "interactions": 0,
            "pending_person_matches": [],
        }

        # 1. Action items (graph nodes)
        try:
            items = self.action_extractor.extract(content)
            if items:
                await self._persist_action_items(items, message_id, epistemic)
                summary["action_items"] = len(items)
        except Exception as e:
            logger.warning("Action item routing failed: %s", e)

        # 2. Decisions (graph nodes)
        try:
            decisions = self.decision_extractor.extract(content)
            if decisions:
                await self._persist_decisions(decisions, message_id, epistemic)
                summary["decisions"] = len(decisions)
        except Exception as e:
            logger.warning("Decision routing failed: %s", e)

        # 3. Interaction enrichment (edge properties) + person resolution.
        # Operates on person-type concepts the linker already created.
        try:
            enriched, pending = await self._enrich_person_interactions(
                concepts, content, message_id
            )
            summary["interactions"] = enriched
            summary["pending_person_matches"] = pending
        except Exception as e:
            logger.warning("Interaction enrichment failed: %s", e)

        return summary

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    async def _persist_action_items(
        self,
        items: List[str],
        message_id: Optional[str],
        epistemic: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Idempotent action item persistence as graph nodes.

        Deterministic node_id from (agent, message, text) means reprocessing
        the same message upserts the same node — the graph's
        INSERT OR REPLACE on node_id guarantees at-most-one node per
        (message, text). No separate table, no separate migration.

        When ``epistemic`` is provided (from the message's MemoryMetadata),
        claim_source / claim_certainty / temporal_validity are stored on the
        node so epistemic provenance flows from message to claim.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for text in items:
            node_id = _deterministic_action_node_id(self.agent_id, message_id, text)

            # Preserve existing status / assignee / due_date if the node
            # already exists — an earlier extraction might have been
            # updated by the user (marked done, assigned to a person).
            existing = await self.graph.get_node(node_id)
            existing_props = (existing.properties if existing else {}) or {}

            properties = {
                "text": text,
                "status": existing_props.get("status", "pending"),
                "assignee_concept_id": existing_props.get("assignee_concept_id"),
                "due_date": existing_props.get("due_date"),
                "confidence": 0.7,
                "source_message_id": message_id,
                "agent_id": self.agent_id,
                "created_at": existing_props.get("created_at", now_iso),
                "updated_at": now_iso,
            }
            # Epistemic provenance from message metadata (#670 fix #2)
            if epistemic:
                for key in ("claim_source", "claim_certainty", "temporal_validity"):
                    if key in epistemic:
                        properties[key] = epistemic[key]
            await self.graph.add_node(GraphNode(
                node_id=node_id,
                node_type=ACTION_ITEM_NODE_TYPE,
                label=text[:120],
                properties=properties,
            ))
            if message_id:
                source = f"message:{self.agent_id}:{message_id}"
                await self.graph.add_edge(source, node_id, "records_action")

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    async def _persist_decisions(
        self,
        decisions: List[str],
        message_id: Optional[str],
        epistemic: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Idempotent decision node creation.

        Uses a deterministic node id from (agent, message, text) so that
        reprocessing the same message upserts the same node instead of
        creating parallel nodes. Graph upsert semantics (INSERT OR REPLACE
        on node_id) guarantee at-most-one node per (message, decision text).

        When ``epistemic`` is provided (from the message's MemoryMetadata),
        claim_source / claim_certainty / temporal_validity are stored on the
        node so epistemic provenance flows from message to claim.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for text in decisions:
            node_id = _deterministic_decision_node_id(self.agent_id, message_id, text)
            properties: Dict[str, Any] = {
                "text": text,
                "source_message_id": message_id,
                "confidence": 0.7,
                "created_at": now_iso,
                "agent_id": self.agent_id,
            }
            # Epistemic provenance from message metadata (#670 fix #2)
            if epistemic:
                for key in ("claim_source", "claim_certainty", "temporal_validity"):
                    if key in epistemic:
                        properties[key] = epistemic[key]
            await self.graph.add_node(GraphNode(
                node_id=node_id,
                node_type=DECISION_NODE_TYPE,
                label=text[:120],
                properties=properties,
            ))
            if message_id:
                source = f"message:{self.agent_id}:{message_id}"
                await self.graph.add_edge(source, node_id, "records_decision")

    # ------------------------------------------------------------------
    # Interactions (enrich existing mentions edges) + person resolution
    # ------------------------------------------------------------------

    async def _enrich_person_interactions(
        self,
        concepts: List[str],
        content: str,
        message_id: Optional[str],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Enrich message→person mentions edges with sentiment + topics.

        Also runs person resolution against each person concept; ambiguous
        matches are returned so callers can surface a confirmation UI.
        """
        if not message_id:
            return 0, []

        sentiment, topics = extract_interaction_sentiment(content)
        enriched_count = 0
        pending: List[Dict[str, Any]] = []

        message_node = f"message:{self.agent_id}:{message_id}"

        for concept_label in concepts:
            if not _looks_like_person(concept_label):
                continue
            concept_node_id = f"concept:{self.agent_id}:{concept_label}"

            # Attach interaction properties to the existing mentions edge.
            properties = {
                "sentiment": sentiment,
                "topics": topics,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.graph.add_edge(
                message_node, concept_node_id, "mentions", properties=properties
            )
            enriched_count += 1

            # 3-pass resolution against other person concepts of this agent.
            match = await self.person_resolver.resolve(concept_label, self.agent_id)
            if match.status == "pending":
                pending.append({
                    "mentioned_label": concept_label,
                    "candidates": match.candidates,
                    "message_id": message_id,
                })

        return enriched_count, pending


# =============================================================================
# Supersession
# =============================================================================

# Node types that represent claims and can be superseded.
CLAIM_NODE_TYPES = frozenset({ACTION_ITEM_NODE_TYPE, DECISION_NODE_TYPE})


async def mark_superseded(
    graph: AsyncGraphStore,
    old_node_id: str,
    new_node_id: str,
    reason: str = "",
) -> Dict[str, Any]:
    """Mark a claim node as superseded by another.

    Only ``decision`` and ``action_item`` nodes can be superseded. Attempting
    to supersede any other node type (e.g. a concept or skill) returns a
    structured error.

    Creates a ``supersedes`` edge from *new_node* → *old_node* (direction:
    "new supersedes old"). Updates the old node's properties:

    - ``superseded_by``: the id of the new node
    - ``contradicted_by``: list of node ids that contradict this node.
      The new node is appended to the list. The field name explicitly
      encodes the direction: "this node is contradicted BY those nodes."

    Args:
        graph: The async graph store.
        old_node_id: Node being replaced.
        new_node_id: Node that replaces it.
        reason: Human-readable reason for supersession.

    Returns:
        ``{"success": True, ...}`` on success, or
        ``{"success": False, "error": ..., "error_code": ...}`` on failure.
    """
    old_node = await graph.get_node(old_node_id)
    if old_node is None:
        return {
            "success": False,
            "error": f"Node {old_node_id} not found",
            "error_code": "NODE_NOT_FOUND",
        }

    if old_node.node_type not in CLAIM_NODE_TYPES:
        return {
            "success": False,
            "error": (
                f"Cannot supersede node of type '{old_node.node_type}'. "
                f"Only claim nodes ({', '.join(sorted(CLAIM_NODE_TYPES))}) "
                f"can be superseded."
            ),
            "error_code": "INVALID_NODE_TYPE",
            "node_type": old_node.node_type,
        }

    new_node = await graph.get_node(new_node_id)
    if new_node is None:
        return {
            "success": False,
            "error": f"Node {new_node_id} not found",
            "error_code": "NODE_NOT_FOUND",
        }

    if new_node.node_type not in CLAIM_NODE_TYPES:
        return {
            "success": False,
            "error": (
                f"Superseding node must be a claim type, got '{new_node.node_type}'."
            ),
            "error_code": "INVALID_NODE_TYPE",
            "node_type": new_node.node_type,
        }

    # Update old node properties.
    props = dict(old_node.properties or {})
    props["superseded_by"] = new_node_id
    contradicted_by = props.get("contradicted_by") or []
    if new_node_id not in contradicted_by:
        contradicted_by.append(new_node_id)
    props["contradicted_by"] = contradicted_by

    await graph.add_node(GraphNode(
        node_id=old_node.node_id,
        node_type=old_node.node_type,
        label=old_node.label,
        properties=props,
    ))

    # Edge: new → old, label "supersedes"
    edge_props: Dict[str, Any] = {"reason": reason}
    await graph.add_edge(new_node_id, old_node_id, "supersedes", properties=edge_props)

    return {
        "success": True,
        "old_node_id": old_node_id,
        "new_node_id": new_node_id,
        "superseded_by": new_node_id,
    }


# =============================================================================
# Helpers
# =============================================================================


# Keywords from the existing AssociativeLinker that indicate person concepts.
# Duplicated here rather than imported so a later redesign of the linker's
# category taxonomy doesn't accidentally silence the router.
_PERSON_KEYWORDS = frozenset([
    "mom", "mother", "mama", "mommy", "dad", "father", "papa", "daddy",
    "wife", "husband", "spouse", "partner", "son", "daughter", "child",
    "kid", "baby", "brother", "sister", "sibling", "grandma", "grandmother",
    "nana", "granny", "grandpa", "grandfather", "gramps", "friend", "buddy",
    "bestie", "pal", "boss", "manager", "coworker", "colleague", "doctor",
    "therapist", "counselor",
])


# Concepts the linker produces that are definitely NOT people — covers the
# linker's place/time/activity/emotion categories so we don't mis-enrich
# mentions edges as interactions.
_NON_PERSON_KEYWORDS = frozenset([
    # Places
    "home", "house", "apartment", "place", "work", "office", "job",
    "workplace", "school", "college", "university", "class", "hospital",
    "clinic", "church", "temple", "mosque", "synagogue",
    # Time
    "morning", "afternoon", "evening", "night", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "christmas", "thanksgiving", "birthday", "anniversary", "holiday",
    "childhood", "teenager", "adult", "elderly", "young",
    # Activities
    "cooking", "baking", "gardening", "reading", "writing",
    "running", "walking", "exercise", "workout", "gym",
    "music", "singing", "dancing", "playing",
    "travel", "vacation", "trip", "visit",
    "meeting", "project", "deadline",
    # Emotions
    "happy", "sad", "angry", "scared", "anxious", "excited",
    "love", "hate", "miss", "worry", "fear",
    "stress", "peace", "calm", "chaos",
])


def _deterministic_decision_node_id(
    agent_id: str,
    message_id: Optional[str],
    text: str,
) -> str:
    """Stable node id for decision nodes. Same input → same id → upsert."""
    if message_id is None:
        return f"decision:{agent_id}:{uuid.uuid4().hex[:12]}"
    digest = hashlib.sha1(
        f"{message_id}\x00{text.strip().lower()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"decision:{agent_id}:{digest}"


def _deterministic_action_node_id(
    agent_id: str,
    message_id: Optional[str],
    text: str,
) -> str:
    """Stable node id for action item nodes."""
    if message_id is None:
        return f"action:{agent_id}:{uuid.uuid4().hex[:12]}"
    digest = hashlib.sha1(
        f"{message_id}\x00{text.strip().lower()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"action:{agent_id}:{digest}"


def _looks_like_person(concept: str) -> bool:
    """Heuristic: is this concept label likely a person reference?

    The AssociativeLinker does not annotate concepts with their category,
    so we reconstruct the category by matching against the linker's
    keyword sets. A cleaner fix would be to have the linker emit typed
    concepts; deferred to a follow-up so this PR doesn't sprawl.
    """
    if not concept:
        return False
    lower = concept.lower()
    if lower in _PERSON_KEYWORDS:
        return True
    if lower in _NON_PERSON_KEYWORDS:
        return False
    # Otherwise: treat alphabetic multi-character concepts as probable
    # proper names. Concepts from the linker are lowercased but proper
    # nouns still end up here via the capitalized-word extraction path.
    return len(concept) >= 3 and concept.isalpha()
