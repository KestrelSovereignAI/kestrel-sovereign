"""
Concept association using existing GraphStore.

Builds concept associations in the knowledge graph to enable
human-like associative memory recall. When one concept is mentioned,
related concepts can be activated.

Example: "Mom" triggers "Sunday calls", "Brooklyn", "her garden"
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Dict, Any
from datetime import datetime, timezone

from .async_graph_store import AsyncGraphStore, GraphNode

logger = logging.getLogger(__name__)


ConceptCategory = Literal[
    "person", "place", "time", "activity", "emotion", "proper_noun"
]


#: A capitalised stop word ends a proper-noun run: "Jon And Doe" is not one name.
_RUN_STOP_WORDS = frozenset({"the", "and", "but", "for"})


def classify_label(label: str) -> Optional[str]:
    """The keyword category the linker would give ``label`` on its own, or
    ``None`` when no keyword pass claims it (a proper noun, or nothing).

    The same pattern tables the extraction uses, run over the label alone,
    so a concept node written before categories were stored can be told
    apart on read: "march" is ``time``, "brooklyn" is ``place``, "mom" is
    ``person``, "alice" is ``None`` (#3259).
    """
    text = str(label or "").strip().lower()
    if not text:
        return None
    for category, patterns in (
        ("person", AssociativeLinker.PERSON_PATTERNS),
        ("place", AssociativeLinker.PLACE_PATTERNS),
        ("time", AssociativeLinker.TIME_PATTERNS),
        ("activity", AssociativeLinker.ACTIVITY_PATTERNS),
        ("emotion", AssociativeLinker.EMOTION_PATTERNS),
    ):
        for pattern in patterns:
            if re.search(pattern, text, re.I):
                return category
    return None

@dataclass
class LinkedConcept:
    """A concept extracted by AssociativeLinker with its graph node ID and category.

    This is the contract between the linker and downstream consumers
    (e.g. SchemaRouter). Consumers should use ``node_id`` directly
    instead of reconstructing it from the label.
    """
    node_id: str
    label: str
    category: ConceptCategory


class AssociativeLinker:
    """
    Builds concept associations in the knowledge graph.

    Extracts concepts from messages and creates/strengthens links
    between co-occurring concepts. This enables associative recall:
    when one concept is mentioned, related concepts are surfaced.
    """

    # ─────────────────────────────────────────────────────────────────
    # Concept Extraction Patterns
    # ─────────────────────────────────────────────────────────────────

    # Person relationships (high-value concepts)
    PERSON_PATTERNS = [
        r"\b(mom|mother|mama|mommy)\b",
        r"\b(dad|father|papa|daddy)\b",
        r"\b(wife|husband|spouse|partner)\b",
        r"\b(son|daughter|child|kid|baby)\b",
        r"\b(brother|sister|sibling)\b",
        r"\b(grandma|grandmother|nana|granny)\b",
        r"\b(grandpa|grandfather|papa|gramps)\b",
        r"\b(friend|buddy|bestie|pal)\b",
        r"\b(boss|manager|coworker|colleague)\b",
        r"\b(doctor|therapist|counselor)\b",
    ]

    # Places (context concepts)
    PLACE_PATTERNS = [
        r"\b(home|house|apartment|place)\b",
        r"\b(work|office|job|workplace)\b",
        r"\b(school|college|university|class)\b",
        r"\b(hospital|clinic|doctor'?s)\b",
        r"\b(church|temple|mosque|synagogue)\b",
        # Named places (capitalized)
        r"\b([A-Z][a-z]+ City)\b",
        r"\b(New York|Los Angeles|Chicago|Houston|Brooklyn|Manhattan)\b",
    ]

    # Time concepts (temporal context)
    TIME_PATTERNS = [
        r"\b(morning|afternoon|evening|night)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(christmas|thanksgiving|birthday|anniversary|holiday)\b",
        r"\b(childhood|teenager|adult|elderly|young)\b",
    ]

    # Activity concepts
    ACTIVITY_PATTERNS = [
        r"\b(cooking|baking|gardening|reading|writing)\b",
        r"\b(running|walking|exercise|workout|gym)\b",
        r"\b(music|singing|dancing|playing)\b",
        r"\b(travel|vacation|trip|visit)\b",
        r"\b(work|meeting|project|deadline)\b",
    ]

    # Emotional concepts
    EMOTION_PATTERNS = [
        r"\b(happy|sad|angry|scared|anxious|excited)\b",
        r"\b(love|hate|miss|worry|fear)\b",
        r"\b(stress|peace|calm|chaos)\b",
    ]

    def __init__(self, graph: AsyncGraphStore):
        """
        Initialize with graph store.

        Args:
            graph: AsyncGraphStore instance for concept storage
        """
        self.graph = graph

    async def extract_and_link(
        self,
        message_id: str,
        content: str,
        agent_id: str
    ) -> List[LinkedConcept]:
        """
        Extract concepts from message and create graph links.

        Args:
            message_id: Unique ID of the message
            content: Message text to analyze
            agent_id: Agent ID for scoping

        Returns:
            List of LinkedConcept objects with node_id, label, and category
        """
        categorized = self._extract_concepts_categorized(content)

        # The message node is the durable source for schema-routed action and
        # decision edges too, not only for concept mentions.  Create it before
        # the empty-concept return so the downstream router never leaves an
        # orphan typed node when a commitment contains no recognized concept.
        message_node_id = f"message:{agent_id}:{message_id}"
        await self._ensure_message_node(message_node_id, message_id, agent_id)

        if not categorized:
            return []

        labels = [label for label, _ in categorized]

        # Create/update concept nodes, stamping the category the extraction
        # gave each one: the person resolver reads it to keep months and
        # places out of a person's candidate list (#3259).
        for label, category in categorized:
            await self._ensure_concept_node(label, agent_id, category)

        # Create message → concept links
        linked: List[LinkedConcept] = []
        for label, category in categorized:
            concept_node_id = f"concept:{agent_id}:{label}"
            await self.graph.add_edge(
                message_node_id,
                concept_node_id,
                "mentions"
            )
            linked.append(LinkedConcept(
                node_id=concept_node_id,
                label=label,
                category=category,
            ))

        # Strengthen co-occurring concept associations
        await self._strengthen_cooccurrences(labels, agent_id)

        logger.debug(f"Extracted {len(linked)} concepts: {[c.label for c in linked]}")
        return linked

    def _extract_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts from text (bare-string form).

        Backward-compatible helper that returns only labels.
        Prefer ``_extract_concepts_categorized`` for typed results.

        Returns:
            List of normalized concept strings (lowercase)
        """
        return [label for label, _ in self._extract_concepts_categorized(content)]

    def _extract_concepts_categorized(
        self, content: str
    ) -> List[tuple[str, ConceptCategory]]:
        """
        Extract key concepts from text with their categories.

        Returns:
            List of (normalized_label, category) tuples
        """
        seen: Set[str] = set()
        results: List[tuple[str, ConceptCategory]] = []
        content_lower = content.lower()

        category_patterns: List[tuple[ConceptCategory, List[str]]] = [
            ("person", self.PERSON_PATTERNS),
            ("place", self.PLACE_PATTERNS),
            ("time", self.TIME_PATTERNS),
            ("activity", self.ACTIVITY_PATTERNS),
            ("emotion", self.EMOTION_PATTERNS),
        ]

        for category, patterns in category_patterns:
            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.I)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    normalized = match.lower().strip()
                    if len(normalized) >= 2 and normalized not in seen:
                        seen.add(normalized)
                        results.append((normalized, category))

        # Also extract proper nouns. A run of consecutive capitalized words is
        # ONE name ("Jon Doe"), not one concept per token (#3259). A word that
        # starts a sentence is capitalized for being first and is never part
        # of a name: "Thanks Jon" and "Jon Doe" are the same shape there, and
        # without a lexicon the honest reading is the existing one, so a
        # sentence-initial name loses its first token ("Jon Doe helped" ->
        # "doe"); put the name mid-sentence to keep it whole.
        words = content.split()
        i = 0
        while i < len(words):
            if i == 0 or words[i - 1][-1] in ".!?":
                i += 1
                continue
            run: List[str] = []
            j = i
            while j < len(words) and words[j][0].isupper() and len(words[j]) > 2:
                token = re.sub(r"[^\w]", "", words[j]).lower()
                # A word the keyword passes already classified ("Monday",
                # "Christmas") is its own concept, never part of a name:
                # "Robert Monday" is Robert, on Monday. A stop word ends a
                # run the same way.
                if token in seen or token in _RUN_STOP_WORDS:
                    break
                run.append(words[j])
                if words[j][-1] in ".!?,;:":
                    j += 1
                    break
                j += 1
            if run:
                clean = " ".join(
                    part for part in (re.sub(r"[^\w]", "", w).lower() for w in run) if part
                )
                if clean and clean not in seen:
                    seen.add(clean)
                    results.append((clean, "proper_noun"))
            i = max(j, i + 1)

        return results

    async def _ensure_concept_node(
        self,
        concept: str,
        agent_id: str,
        category: Optional[str] = None,
    ) -> None:
        """Create or update concept node in graph.

        ``category`` is recorded on the node (and refreshed on every
        mention, so a node written before categories were stored acquires
        one the next time it is mentioned).
        """
        concept_node_id = f"concept:{agent_id}:{concept}"

        existing = await self.graph.get_node(concept_node_id)
        if existing:
            # Update mention count
            props = existing.properties or {}
            props["mention_count"] = props.get("mention_count", 0) + 1
            props["last_mentioned"] = datetime.now(timezone.utc).isoformat()
            if category:
                props["category"] = category
            await self.graph.add_node(GraphNode(
                node_id=concept_node_id,
                node_type="concept",
                label=concept,
                properties=props,
            ))
        else:
            # Create new concept node
            await self.graph.add_node(GraphNode(
                node_id=concept_node_id,
                node_type="concept",
                label=concept,
                properties={
                    "mention_count": 1,
                    "agent_id": agent_id,
                    "first_mentioned": datetime.now(timezone.utc).isoformat(),
                    "last_mentioned": datetime.now(timezone.utc).isoformat(),
                    **({"category": category} if category else {}),
                },
            ))

    async def _ensure_message_node(
        self,
        node_id: str,
        message_id: str,
        agent_id: str,
    ) -> None:
        """Create message node if it doesn't exist."""
        existing = await self.graph.get_node(node_id)
        if not existing:
            await self.graph.add_node(GraphNode(
                node_id=node_id,
                node_type="message",
                label=f"Message {message_id}",
                properties={
                    "message_id": message_id,
                    "agent_id": agent_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            ))

    async def _strengthen_cooccurrences(
        self,
        concepts: List[str],
        agent_id: str
    ) -> None:
        """
        Strengthen associations between co-occurring concepts.

        When concepts appear together in a message, they become
        more strongly associated. This is the heart of associative memory.
        """
        if len(concepts) < 2:
            return

        # Create/strengthen edges between all pairs
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                await self._strengthen_association(c1, c2, agent_id)

    async def _strengthen_association(
        self,
        concept1: str,
        concept2: str,
        agent_id: str,
        boost: float = 0.1
    ) -> None:
        """Strengthen association between two concepts."""
        # Ensure consistent ordering for edge lookup
        if concept1 > concept2:
            concept1, concept2 = concept2, concept1

        node1_id = f"concept:{agent_id}:{concept1}"
        node2_id = f"concept:{agent_id}:{concept2}"

        # Get existing edges
        edges = await self.graph.get_edges(node1_id, direction="out")

        existing_edge = None
        for edge in edges:
            if edge.target_id == node2_id and edge.label == "associated_with":
                existing_edge = edge
                break

        if existing_edge:
            # Strengthen existing association
            props = existing_edge.properties or {}
            current_strength = props.get("strength", 0.0)
            props["strength"] = min(1.0, current_strength + boost)
            props["last_cooccurrence"] = datetime.now(timezone.utc).isoformat()
            props["cooccurrence_count"] = props.get("cooccurrence_count", 0) + 1
            await self.graph.add_edge(
                node1_id,
                node2_id,
                "associated_with",
                props
            )
        else:
            # Create new association
            await self.graph.add_edge(
                node1_id,
                node2_id,
                "associated_with",
                {
                    "strength": boost,
                    "first_cooccurrence": datetime.now(timezone.utc).isoformat(),
                    "last_cooccurrence": datetime.now(timezone.utc).isoformat(),
                    "cooccurrence_count": 1,
                }
            )

    async def get_associated_concepts(
        self,
        concept: str,
        agent_id: str,
        min_strength: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get concepts associated with given concept.

        Args:
            concept: The concept to find associations for
            agent_id: Agent ID for scoping
            min_strength: Minimum association strength (0.0 to 1.0)

        Returns:
            List of dicts with 'concept' and 'strength' keys,
            sorted by strength descending
        """
        node_id = f"concept:{agent_id}:{concept}"

        # Get edges in both directions (associations are bidirectional)
        out_edges = await self.graph.get_edges(node_id, direction="out")
        in_edges = await self.graph.get_edges(node_id, direction="in")

        associated = []

        for edge in out_edges:
            if edge.label == "associated_with":
                props = edge.properties or {}
                strength = props.get("strength", 0.0)
                if strength >= min_strength:
                    # Extract concept name from node_id
                    parts = edge.target_id.split(":")
                    if len(parts) >= 3:
                        associated.append({
                            "concept": parts[-1],
                            "strength": strength,
                            "cooccurrence_count": props.get("cooccurrence_count", 0),
                        })

        for edge in in_edges:
            if edge.label == "associated_with":
                props = edge.properties or {}
                strength = props.get("strength", 0.0)
                if strength >= min_strength:
                    parts = edge.source_id.split(":")
                    if len(parts) >= 3:
                        associated.append({
                            "concept": parts[-1],
                            "strength": strength,
                            "cooccurrence_count": props.get("cooccurrence_count", 0),
                        })

        # Sort by strength descending
        associated.sort(key=lambda x: x["strength"], reverse=True)

        return associated

    async def get_concept_network(
        self,
        concept: str,
        agent_id: str,
        depth: int = 2,
        min_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Get network of concepts around a central concept.

        Args:
            concept: Central concept
            agent_id: Agent ID
            depth: How many hops to explore
            min_strength: Minimum edge strength to follow

        Returns:
            Dict with 'nodes' and 'edges' for visualization
        """
        visited: Set[str] = set()
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        async def explore(c: str, current_depth: int):
            if c in visited or current_depth > depth:
                return
            visited.add(c)

            # Get concept node
            node_id = f"concept:{agent_id}:{c}"
            node = await self.graph.get_node(node_id)
            if node:
                nodes.append({
                    "id": c,
                    "label": c,
                    "mention_count": node.properties.get("mention_count", 0),
                })

            # Get associations
            associated = await self.get_associated_concepts(
                c, agent_id, min_strength
            )

            for assoc in associated:
                target = assoc["concept"]
                if target not in visited:
                    edges.append({
                        "source": c,
                        "target": target,
                        "strength": assoc["strength"],
                    })
                    await explore(target, current_depth + 1)

        await explore(concept, 0)

        return {
            "center": concept,
            "nodes": nodes,
            "edges": edges,
        }

    async def find_concepts_for_query(
        self,
        query: str,
        agent_id: str
    ) -> List[str]:
        """
        Find all concepts relevant to a query.

        Extracts concepts from query, then expands with associations.

        Args:
            query: Search query text
            agent_id: Agent ID

        Returns:
            List of concept strings (original + associated)
        """
        # Extract concepts from query
        direct_concepts = self._extract_concepts(query)

        # Expand with associations
        all_concepts = set(direct_concepts)

        for concept in direct_concepts:
            associated = await self.get_associated_concepts(
                concept, agent_id, min_strength=0.2
            )
            for assoc in associated[:5]:  # Top 5 associations
                all_concepts.add(assoc["concept"])

        return list(all_concepts)
