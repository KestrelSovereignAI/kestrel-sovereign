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
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .associative_linker import LinkedConcept
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


# ---------------------------------------------------------------------------
# Extraction precision guards (#2852)
# ---------------------------------------------------------------------------
# The capture class in every pattern above is ``[^.!?\n]+``, which stops BEFORE
# terminal punctuation. An interrogative therefore reads exactly like a
# commitment: "do I need a heartbeat agent or what?" captures as the action
# item "a heartbeat agent or what". Both false positives observed in production
# were this shape. Nothing downstream can recover the distinction, because by
# then the "?" has been discarded — so it is checked here, against the
# enclosing sentence rather than the captured span.

_SENTENCE_BOUNDARIES = (".", "!", "?", "\n")

# A negated clause is not a commitment.
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|won'?t|don'?t|doesn'?t|didn'?t|can'?t|"
    r"cannot|shouldn'?t|wouldn'?t)\b",
    re.IGNORECASE,
)

# Explicit task directives. These stay tasks even when phrased as questions.
_EXPLICIT_CUE_RE = re.compile(
    r"\bTODO:?|\bremind me to\b|\bdon'?t forget to\b", re.IGNORECASE
)

# Reminder cues that CONTAIN a negation token but are positive commitments.
_POSITIVE_CUE_RE = re.compile(
    r"\b(?:don'?t|never)\s+forget\s+to\b|\bremind me to\b", re.IGNORECASE
)

# Trailing filler marks the clause as musing rather than committing.
_FILLER_TAIL_RE = re.compile(
    r"\b(?:or\s+(?:what|something|whatever|anything)|i\s+guess|i\s+suppose|"
    r"maybe|perhaps)\s*$",
    re.IGNORECASE,
)


def _enclosing_sentence(content: str, start: int, end: int) -> str:
    """The sentence containing ``content[start:end]``, terminator included."""
    left = max(content.rfind(c, 0, start) for c in _SENTENCE_BOUNDARIES)
    rights = [i for i in (content.find(c, end) for c in _SENTENCE_BOUNDARIES) if i != -1]
    right = min(rights) if rights else len(content) - 1
    return content[left + 1: right + 1]


def _sentence_offset(content: str, match, sentence: str) -> int:
    """Index in ``content`` where ``sentence`` begins."""
    return content.rfind(sentence, 0, match.end()) if sentence else 0


def _is_committing_clause(
    content: str, match, text: str, *, negative_action_ok: bool = False
) -> bool:
    """Whether a raw pattern hit is a real commitment worth persisting.

    Rejects the three shapes that produced the observed false positives: an
    interrogative sentence, a negated commitment, and a clause trailing off
    into filler.

    ``negative_action_ok`` distinguishes the two consumers. For an ACTION ITEM
    a negated action is no task at all ("I should not restart"). For a
    DECISION it is still a decision — choosing *not* to do something is an
    explicit choice, and "We've decided not to deploy" must be recorded.
    """
    matched = match.group(0)
    sentence = _enclosing_sentence(content, match.start(), match.end())
    explicit_cue = bool(_EXPLICIT_CUE_RE.search(matched))

    # Interrogation is a property of the whole sentence — but only for
    # SPECULATIVE commitments. An explicit directive stays a task even when its
    # action is phrased as a question: "TODO: determine why CI is failing?" and
    # "Remind me to ask whether the deploy succeeded?" are real tasks, and the
    # base extractor kept them (codex review r3).
    if not explicit_cue and sentence.rstrip().endswith("?"):
        return False

    # Negation attaches to the COMMITMENT OPERATOR, not to the sentence and not
    # to the action. Scoping it anywhere wider regresses real extractions
    # (codex review r1/r2):
    #   * sentence-wide  -> "I don't need to deploy, but I will restart the
    #                       host" loses the genuine second commitment;
    #   * whole match    -> "I need to ensure the backup never expires" is
    #                       dropped because the ACTION contains "never", and
    #                       "We've decided not to deploy" is dropped even
    #                       though choosing not to do something is a decision.
    # The operator is the matched span with the captured action removed.
    operator = matched[: match.start(match.lastindex or 0) - match.start()] if match.lastindex else matched

    # The operator alone is not enough. Two shapes slip past it (codex r3):
    #   * "I should not restart"        -> the capture STARTS at "not", so the
    #                                      operator is clean and "not restart"
    #                                      would persist as an action;
    #   * "I don't think I need to X"   -> the match begins at the INNER cue,
    #                                      so the negator sits just before it.
    # Widen to the leading words of the action, and to a short look-back that
    # stops at a conjunction so an unrelated negated clause can't veto.
    action_head = " ".join(text.split()[:2])
    lookback = sentence[: max(0, match.start() - _sentence_offset(content, match, sentence))]
    lookback = re.split(r"\b(?:but|and|however|though)\b|[,;]", lookback)[-1]

    negated = _NEGATION_RE.search(operator) or _NEGATION_RE.search(lookback)
    if not negative_action_ok:
        negated = negated or _NEGATION_RE.search(action_head)

    if not _POSITIVE_CUE_RE.search(operator) and negated:
        return False

    if _FILLER_TAIL_RE.search(text):
        return False
    return True


# Evidence strength per claim. Confidence was previously the literal 0.7 on
# every extracted item and decision, so the field carried no information and no
# threshold could be tuned against it (#2852). An explicit ``TODO:`` is far
# stronger evidence of a commitment than a bare "I'll".
_STRONG_EVIDENCE_RE = re.compile(
    r"\bTODO:?|\bremind me to\b|\bdon'?t forget to\b", re.IGNORECASE
)
STRONG_CLAIM_CONFIDENCE = 0.9
DEFAULT_CLAIM_CONFIDENCE = 0.7


def claim_confidence(text: str, source: str = "") -> float:
    """Confidence for one extracted claim, derived from its evidence."""
    probe = source or text
    return (
        STRONG_CLAIM_CONFIDENCE
        if _STRONG_EVIDENCE_RE.search(probe)
        else DEFAULT_CLAIM_CONFIDENCE
    )


class ActionItemExtractor:
    """Regex-based action item extraction.

    Intentionally simple — captures first-person commitments, TODOs, and
    reminders. An LLM-powered extractor can replace this later without
    changing the surrounding routing code.
    """

    def extract(self, content: str) -> List[str]:
        return [text for text, _evidence in self.extract_with_evidence(content)]

    def extract_with_evidence(self, content: str) -> List[Tuple[str, str]]:
        """Extracted items paired with the raw span that matched.

        The captured text has the cue stripped ("TODO: ship it" -> "ship it"),
        so confidence scored from the capture alone can never see the strongest
        evidence there is (codex review r1). Callers that persist confidence
        must use this, not :meth:`extract`.
        """
        items: List[Tuple[str, str]] = []
        seen: Dict[str, int] = {}
        for pattern in ACTION_ITEM_PATTERNS:
            for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                text = match.group(len(match.groups())).strip().strip(",;")
                if not text or len(text) < 3:
                    continue
                if not _is_committing_clause(content, match, text):
                    continue
                key = text.lower()
                evidence = match.group(0)
                if key in seen:
                    # Same action, different cue. "I will ship. TODO: ship."
                    # matches the weak first-person pattern first, so keeping
                    # the first sighting would persist 0.7 despite the explicit
                    # TODO (codex review r2). Upgrade in place instead.
                    existing = seen[key]
                    if claim_confidence(text, evidence) > claim_confidence(
                        text, items[existing][1]
                    ):
                        items[existing] = (text, evidence)
                    continue
                seen[key] = len(items)
                items.append((text, evidence))
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
                # Deciding NOT to do something is still a decision.
                if not _is_committing_clause(
                    content, match, text, negative_action_ok=True
                ):
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
        concepts: List[LinkedConcept],
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route extracted structure from a message.

        Args:
            concepts: Typed concept objects from AssociativeLinker.
                Each carries its graph node_id and category so the
                router never needs to reconstruct IDs or guess categories.
            metadata: Enriched message metadata from EmotionalTagger.
                Epistemic fields (claim_certainty, claim_source,
                temporal_validity) are inherited by claim-shaped nodes.

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

        # Extract epistemic provenance from message metadata for claim nodes
        epistemic = _extract_epistemic_fields(metadata)

        # 1. Action items (graph nodes)
        try:
            items = self.action_extractor.extract_with_evidence(content)
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
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for item in items:
            # Items arrive as (text, matched-span) so confidence can see the
            # cue that extraction stripped (codex review r1). Tolerate a bare
            # string for any caller still on the older shape.
            text, evidence = item if isinstance(item, tuple) else (item, "")
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
                "confidence": claim_confidence(text, evidence),
                "source_message_id": message_id,
                "agent_id": self.agent_id,
                "created_at": existing_props.get("created_at", now_iso),
                "updated_at": now_iso,
            }
            # Inherit epistemic provenance from message metadata
            if epistemic:
                for key in ("claim_certainty", "claim_source", "temporal_validity"):
                    if epistemic.get(key) is not None:
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
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for text in decisions:
            node_id = _deterministic_decision_node_id(self.agent_id, message_id, text)
            properties = {
                "text": text,
                "source_message_id": message_id,
                "confidence": claim_confidence(text),
                "created_at": now_iso,
                "agent_id": self.agent_id,
            }
            # Inherit epistemic provenance from message metadata
            if epistemic:
                for key in ("claim_certainty", "claim_source", "temporal_validity"):
                    if epistemic.get(key) is not None:
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
        concepts: List[LinkedConcept],
        content: str,
        message_id: Optional[str],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Enrich message→person mentions edges with sentiment + topics.

        Uses ``concept.category`` from the linker to identify person
        concepts — no local keyword heuristics needed. Also uses
        ``concept.node_id`` directly instead of reconstructing it.

        Both ``person`` (keyword matches like "mom") and ``proper_noun``
        (capitalized names like "Alice") are treated as enrichable for
        interaction edges.

        Ambiguous person matches are returned so callers can surface a
        confirmation UI.
        """
        if not message_id:
            return 0, []

        sentiment, topics = extract_interaction_sentiment(content)
        enriched_count = 0
        pending: List[Dict[str, Any]] = []

        message_node = f"message:{self.agent_id}:{message_id}"

        _person_categories = {"person", "proper_noun"}
        for concept in concepts:
            if concept.category not in _person_categories:
                continue

            # Attach interaction properties to the existing mentions edge.
            properties = {
                "sentiment": sentiment,
                "topics": topics,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.graph.add_edge(
                message_node, concept.node_id, "mentions", properties=properties
            )
            enriched_count += 1

            # 3-pass resolution against other person concepts of this agent.
            match = await self.person_resolver.resolve(concept.label, self.agent_id)
            if match.status == "pending":
                pending.append({
                    "mentioned_label": concept.label,
                    "candidates": match.candidates,
                    "message_id": message_id,
                })

        return enriched_count, pending


# =============================================================================
# Helpers
# =============================================================================




def _extract_epistemic_fields(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract epistemic provenance fields from enriched message metadata."""
    if not metadata:
        return None
    fields = {}
    for key in ("claim_certainty", "claim_source", "temporal_validity"):
        val = metadata.get(key)
        if val is not None:
            fields[key] = val
    return fields if fields else None


CLAIM_SHAPED_NODE_TYPES = frozenset({ACTION_ITEM_NODE_TYPE, DECISION_NODE_TYPE})


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


