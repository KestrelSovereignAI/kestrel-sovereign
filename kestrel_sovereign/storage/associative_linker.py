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
from typing import List, Set, Dict, Any, Optional
from datetime import datetime, timezone

from .async_graph_store import AsyncGraphStore, GraphNode, Edge


@dataclass(frozen=True)
class LinkedConcept:
    """Typed concept returned by :meth:`AssociativeLinker.extract_and_link`.

    Provides canonical-by-convention node IDs so downstream consumers
    (e.g. SchemaRouter) can reference graph nodes without reconstructing
    the ID format themselves.

    Attributes:
        node_id: Deterministic graph node ID (``concept:{agent_id}:{label}``).
        label: Original extracted text, normalised to lowercase.
        category: One of ``person``, ``place``, ``time``, ``activity``,
            ``emotion``, ``proper_noun``.
    """
    node_id: str
    label: str
    category: str

logger = logging.getLogger(__name__)


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

        The linker provides canonical-by-convention node IDs — downstream
        consumers can reference graph nodes directly without reconstructing
        the deterministic ID format.

        Args:
            message_id: Unique ID of the message
            content: Message text to analyze
            agent_id: Agent ID for scoping

        Returns:
            List of :class:`LinkedConcept` with ``node_id``, ``label``,
            and ``category`` for each extracted concept.
        """
        categorized = self._extract_concepts_categorized(content)

        if not categorized:
            return []

        labels = [label for label, _cat in categorized]

        # Create/update concept nodes
        for label in labels:
            await self._ensure_concept_node(label, agent_id)

        # Create message → concept links
        message_node_id = f"message:{agent_id}:{message_id}"
        await self._ensure_message_node(message_node_id, message_id)

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

        logger.debug(f"Extracted {len(linked)} concepts: {[lc.label for lc in linked]}")
        return linked

    async def extract_and_link_labels(
        self,
        message_id: str,
        content: str,
        agent_id: str
    ) -> List[str]:
        """Backward-compatible wrapper returning bare label strings.

        Callers that only need concept labels (not the full typed data)
        can use this instead of :meth:`extract_and_link`.
        """
        linked = await self.extract_and_link(message_id, content, agent_id)
        return [lc.label for lc in linked]

    # Mapping from pattern list attribute → category name.
    _PATTERN_CATEGORIES = [
        ("PERSON_PATTERNS", "person"),
        ("PLACE_PATTERNS", "place"),
        ("TIME_PATTERNS", "time"),
        ("ACTIVITY_PATTERNS", "activity"),
        ("EMOTION_PATTERNS", "emotion"),
    ]

    def _extract_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts from text (labels only).

        Returns:
            List of normalized concept strings (lowercase)
        """
        return [label for label, _cat in self._extract_concepts_categorized(content)]

    def _extract_concepts_categorized(self, content: str) -> List[tuple]:
        """
        Extract key concepts from text with their category.

        Returns:
            List of ``(label, category)`` tuples. ``label`` is normalised
            to lowercase. ``category`` is one of ``person``, ``place``,
            ``time``, ``activity``, ``emotion``, or ``proper_noun``.
        """
        # Map label → category; first match wins (pattern categories
        # are checked in order, then proper nouns).
        seen: Dict[str, str] = {}
        content_lower = content.lower()

        for attr, category in self._PATTERN_CATEGORIES:
            patterns = getattr(self, attr)
            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.I)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    normalized = match.lower().strip()
                    if len(normalized) >= 2 and normalized not in seen:
                        seen[normalized] = category

        # Also extract proper nouns (capitalized words that aren't sentence starters)
        words = content.split()
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and words[i-1][-1] not in ".!?":
                # Check if capitalized and not common word
                if word[0].isupper() and len(word) > 2:
                    clean = re.sub(r"[^\w]", "", word).lower()
                    if clean and clean not in ["the", "and", "but", "for"]:
                        if clean not in seen:
                            seen[clean] = "proper_noun"

        return list(seen.items())

    async def _ensure_concept_node(
        self,
        concept: str,
        agent_id: str
    ) -> None:
        """Create or update concept node in graph."""
        concept_node_id = f"concept:{agent_id}:{concept}"

        existing = await self.graph.get_node(concept_node_id)
        if existing:
            # Update mention count
            props = existing.properties or {}
            props["mention_count"] = props.get("mention_count", 0) + 1
            props["last_mentioned"] = datetime.now(timezone.utc).isoformat()
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
                },
            ))

    async def _ensure_message_node(
        self,
        node_id: str,
        message_id: str
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
