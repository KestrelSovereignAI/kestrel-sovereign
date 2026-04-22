"""Unit tests for AssociativeLinker typed LinkedConcept return shape.

Covers the contract introduced in #662: ``extract_and_link()`` returns
``List[LinkedConcept]`` where each element carries a ``node_id``,
``label``, and ``category``.

Also includes a real-text regression for person-name categorization
(keyword "mom" vs proper noun "Alice").
"""

from __future__ import annotations

from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.associative_linker import (
    AssociativeLinker,
    ConceptCategory,
    LinkedConcept,
)
from kestrel_sovereign.storage.async_graph_store import GraphNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_ID = "agent-test-1"


def _make_graph_mock() -> AsyncMock:
    """Return an AsyncMock mimicking AsyncGraphStore.

    ``get_node`` returns *None* so every concept is treated as new.
    ``add_node``, ``add_edge`` are plain AsyncMock stubs.
    """
    graph = AsyncMock()
    graph.get_node = AsyncMock(return_value=None)
    graph.add_node = AsyncMock()
    graph.add_edge = AsyncMock()
    graph.get_edges = AsyncMock(return_value=[])
    return graph


# ---------------------------------------------------------------------------
# 1. Typed return-shape tests
# ---------------------------------------------------------------------------


class TestExtractAndLinkTypedReturn:
    """Verify the LinkedConcept contract for extract_and_link()."""

    @pytest_asyncio.fixture
    async def linker(self):
        return AssociativeLinker(_make_graph_mock())

    @pytest.mark.asyncio
    async def test_returns_linked_concept_instances(self, linker):
        """Every element must be a LinkedConcept dataclass."""
        results = await linker.extract_and_link(
            "msg-1", "I called mom yesterday", AGENT_ID
        )
        assert len(results) >= 1
        for concept in results:
            assert isinstance(concept, LinkedConcept)

    @pytest.mark.asyncio
    async def test_node_id_matches_graph_convention(self, linker):
        """node_id must follow concept:{agent_id}:{label} format."""
        results = await linker.extract_and_link(
            "msg-2", "mom is visiting Brooklyn", AGENT_ID
        )
        for concept in results:
            expected_prefix = f"concept:{AGENT_ID}:"
            assert concept.node_id.startswith(expected_prefix), (
                f"node_id '{concept.node_id}' doesn't start with '{expected_prefix}'"
            )
            assert concept.node_id == f"concept:{AGENT_ID}:{concept.label}"

    @pytest.mark.asyncio
    async def test_node_id_matches_graph_write(self, linker):
        """node_id on each LinkedConcept must match the ID actually
        written to the graph via add_node."""
        results = await linker.extract_and_link(
            "msg-3", "I miss grandma and her garden", AGENT_ID
        )
        written_ids = {
            call.args[0].node_id
            for call in linker.graph.add_node.await_args_list
            if hasattr(call.args[0], "node_id")
        }
        for concept in results:
            assert concept.node_id in written_ids, (
                f"LinkedConcept node_id '{concept.node_id}' was not written to the graph"
            )

    @pytest.mark.asyncio
    async def test_label_preserves_extracted_text(self, linker):
        """label must be the normalized (lowercase) extracted text."""
        results = await linker.extract_and_link(
            "msg-4", "I was happy at work", AGENT_ID
        )
        labels = {c.label for c in results}
        assert "happy" in labels
        assert "work" in labels

    @pytest.mark.asyncio
    async def test_category_is_valid_enum_value(self, linker):
        """category must be one of the documented ConceptCategory values."""
        valid = set(get_args(ConceptCategory))
        results = await linker.extract_and_link(
            "msg-5",
            "I was happy when mom visited Brooklyn on Christmas",
            AGENT_ID,
        )
        assert len(results) >= 3  # at least emotion, person, place/time
        for concept in results:
            assert concept.category in valid, (
                f"category '{concept.category}' not in {valid}"
            )

    @pytest.mark.asyncio
    async def test_mentions_edges_written(self, linker):
        """Message→concept 'mentions' edges must still be created."""
        results = await linker.extract_and_link(
            "msg-6", "mom and dad went to work", AGENT_ID
        )
        mentions_edges = [
            call
            for call in linker.graph.add_edge.await_args_list
            if len(call.args) >= 3 and call.args[2] == "mentions"
        ]
        concept_ids_from_edges = {call.args[1] for call in mentions_edges}
        concept_ids_from_return = {c.node_id for c in results}
        assert concept_ids_from_return <= concept_ids_from_edges, (
            "Not every returned concept has a 'mentions' edge"
        )

    @pytest.mark.asyncio
    async def test_empty_message_returns_empty(self, linker):
        """A message with no extractable concepts returns []."""
        results = await linker.extract_and_link(
            "msg-7", "the quick brown fox", AGENT_ID
        )
        assert results == []


# ---------------------------------------------------------------------------
# 2. Person-name categorization regression
# ---------------------------------------------------------------------------


class TestPersonNameCategorization:
    """Real-text regression: keyword persons vs proper-noun persons.

    "mom" matches a PERSON_PATTERN → category "person".
    "Alice" is a capitalized word (not a sentence starter) → category
    "proper_noun".

    SchemaRouter should treat *both* ``person`` and ``proper_noun`` as
    enrichable for interaction nodes. This test documents that expectation.
    """

    @pytest_asyncio.fixture
    async def linker(self):
        return AssociativeLinker(_make_graph_mock())

    @pytest.mark.asyncio
    async def test_mom_categorized_as_person(self, linker):
        results = await linker.extract_and_link(
            "msg-pn-1", "I called mom this morning", AGENT_ID
        )
        mom = next((c for c in results if c.label == "mom"), None)
        assert mom is not None, "Expected 'mom' to be extracted"
        assert mom.category == "person"

    @pytest.mark.asyncio
    async def test_proper_noun_alice_categorized(self, linker):
        """A proper noun like 'Alice' (not matching PERSON_PATTERNS)
        should be categorized as 'proper_noun'."""
        results = await linker.extract_and_link(
            "msg-pn-2", "I talked to Alice about the project", AGENT_ID
        )
        alice = next((c for c in results if c.label == "alice"), None)
        assert alice is not None, "Expected 'Alice' to be extracted as proper noun"
        assert alice.category == "proper_noun"

    @pytest.mark.asyncio
    async def test_both_person_and_proper_noun_in_same_message(self, linker):
        """Both shapes coexist in one message."""
        results = await linker.extract_and_link(
            "msg-pn-3", "I told mom that Alice is coming to visit", AGENT_ID
        )
        by_label = {c.label: c for c in results}
        assert "mom" in by_label
        assert by_label["mom"].category == "person"
        assert "alice" in by_label
        assert by_label["alice"].category == "proper_noun"

    @pytest.mark.asyncio
    async def test_person_keyword_friend_categorized_as_person(self, linker):
        """Another keyword person for completeness."""
        results = await linker.extract_and_link(
            "msg-pn-4", "My friend Sarah came over", AGENT_ID
        )
        friend = next((c for c in results if c.label == "friend"), None)
        assert friend is not None
        assert friend.category == "person"
        sarah = next((c for c in results if c.label == "sarah"), None)
        assert sarah is not None
        assert sarah.category == "proper_noun"
