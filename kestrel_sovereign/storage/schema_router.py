"""Schema-aware routing: promote extracted structure to typed storage.

Inspired by OB1's schema-aware routing pattern. Runs after the existing
emotional/importance/temporal tagging and concept linking, and routes
extracted structure to:

- `action_items` SQL table — state-machine data with date/status/assignee
  queries. A table because SQL indexes matter for "what's pending this
  week" and "what did I promise Alice" query shapes.
- `decision` graph nodes — mirrors the "skill" pattern from #643. The
  graph is already the right home for nodes with associative recall
  value, and decisions are structurally closer to nodes than rows.
- Enriched `mentions` edge properties — sentiment and topics. No new
  table: the existing message→concept mentions edge is already the
  "interaction" record. We just add properties to what's there.

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

DECISION_NODE_TYPE = "decision"

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
    """Promote extracted structure to typed storage.

    Runs after concept linking in the message pipeline. Reads the message
    content plus the concept list the linker produced, and writes:

    - action_items rows
    - decision graph nodes
    - enriched properties on the existing mentions edges

    Never creates new concept nodes — operates on what the linker built.
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
        """Create the action_items table.

        action_items is the only new SQL table — decisions are graph nodes
        and interactions are edge properties.
        """
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                source_message_id TEXT,
                text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                assignee_concept_id TEXT,
                due_date TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL
            )
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_action_items_agent_status
            ON action_items(agent_id, status, due_date)
            """
        )
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_action_items_assignee
            ON action_items(agent_id, assignee_concept_id)
            """
        )

    async def route(
        self,
        message_id: str,
        content: str,
        concepts: List[str],
        role: str = "user",
    ) -> Dict[str, Any]:
        """Route extracted structure from a message.

        Returns a summary dict for inclusion in the message's enriched
        metadata. Best-effort: a failure in one lane doesn't block the
        others — the routing pass is advisory enrichment, not critical
        path for storing the message.
        """
        if role != "user":
            return {"action_items": 0, "decisions": 0, "interactions": 0}

        summary: Dict[str, Any] = {
            "action_items": 0,
            "decisions": 0,
            "interactions": 0,
            "pending_person_matches": [],
        }

        # 1. Action items (table)
        try:
            items = self.action_extractor.extract(content)
            if items:
                await self._persist_action_items(items, message_id)
                summary["action_items"] = len(items)
        except Exception as e:
            logger.warning("Action item routing failed: %s", e)

        # 2. Decisions (graph nodes)
        try:
            decisions = self.decision_extractor.extract(content)
            if decisions:
                await self._persist_decisions(decisions, message_id)
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
    ) -> None:
        """Idempotent insert: skip if an action item for this (message, text)
        is already on file. Reprocessing the same message must not duplicate
        rows — replay/retry/backfill are all real operational scenarios.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for text in items:
            item_id = _deterministic_id("action", message_id, text)
            existing = await self.db.fetchone(
                "SELECT id FROM action_items WHERE id = ? AND agent_id = ?",
                (item_id, self.agent_id),
            )
            if existing:
                continue
            await self.db.execute(
                """
                INSERT INTO action_items
                    (id, agent_id, source_message_id, text, status,
                     assignee_concept_id, due_date, confidence, created_at)
                VALUES (?, ?, ?, ?, 'pending', NULL, NULL, 0.7, ?)
                """,
                (item_id, self.agent_id, message_id, text, now_iso),
            )

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    async def _persist_decisions(
        self,
        decisions: List[str],
        message_id: Optional[str],
    ) -> None:
        """Idempotent decision node creation.

        Uses a deterministic node id from (agent, message, text) so that
        reprocessing the same message upserts the same node instead of
        creating parallel nodes. Graph upsert semantics (INSERT OR REPLACE
        on node_id) guarantee at-most-one node per (message, decision text).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for text in decisions:
            node_id = _deterministic_decision_node_id(self.agent_id, message_id, text)
            await self.graph.add_node(GraphNode(
                node_id=node_id,
                node_type=DECISION_NODE_TYPE,
                label=text[:120],
                properties={
                    "text": text,
                    "source_message_id": message_id,
                    "confidence": 0.7,
                    "created_at": now_iso,
                    "agent_id": self.agent_id,
                },
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


def _deterministic_id(kind: str, message_id: Optional[str], text: str) -> str:
    """Generate a stable id from (kind, message, text) so re-routing the
    same message does not produce duplicate rows.

    Falls back to a random id if there's no message_id — in that case
    the caller is routing out-of-band content and has to accept that
    every call creates a new row.
    """
    if message_id is None:
        return f"{kind}_{uuid.uuid4().hex[:12]}"
    digest = hashlib.sha1(
        f"{message_id}\x00{text.strip().lower()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{kind}_{digest}"


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
