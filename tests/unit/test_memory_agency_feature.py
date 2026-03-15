"""
Tests for the MemoryAgencyFeature.

Verifies:
- Pinning a memory sets decay_protected in metadata
- Releasing a memory clears the pin
- Listing pinned memories returns only active pins
- Pin stats return correct ratios
- Double-pinning is idempotent
- Pinning a nonexistent message returns an error
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class FakeDB:
    """In-memory fake database for testing the memory agency feature."""

    def __init__(self):
        self.messages = {}  # id -> (id, content, metadata_json)
        self.pins = {}      # id -> dict with pin fields
        self._next_id = 1

    def add_message(self, content, metadata=None, agent_id="test-agent"):
        """Add a fake message and return its ID."""
        msg_id = self._next_id
        self._next_id += 1
        meta_json = json.dumps(metadata or {})
        self.messages[msg_id] = {
            "id": msg_id,
            "content": content,
            "metadata": meta_json,
            "agent_id": agent_id,
        }
        return msg_id

    async def execute(self, sql, params=()):
        """Handle CREATE TABLE, UPDATE, and INSERT statements."""
        sql_lower = sql.strip().lower()

        if sql_lower.startswith("create table"):
            return 0

        if sql_lower.startswith("update conversation_history"):
            # UPDATE conversation_history SET metadata = ? WHERE id = ?
            meta_json, msg_id = params
            if msg_id in self.messages:
                self.messages[msg_id]["metadata"] = meta_json
            return 1

        if sql_lower.startswith("update memory_pins"):
            # UPDATE memory_pins SET released_at = ? WHERE message_id = ? AND released_at IS NULL
            released_at, message_id = params
            for pin in self.pins.values():
                if pin["message_id"] == message_id and pin["released_at"] is None:
                    pin["released_at"] = released_at
            return 1

        if "insert into memory_pins" in sql_lower:
            pin_id, message_id, agent_id, reason, pinned_at = params
            self.pins[pin_id] = {
                "id": pin_id,
                "message_id": message_id,
                "agent_id": agent_id,
                "pin_reason": reason,
                "pinned_at": pinned_at,
                "released_at": None,
            }
            return 1

        return 0

    async def fetchone(self, sql, params=()):
        """Handle SELECT queries returning a single row."""
        sql_lower = sql.strip().lower()

        if "from conversation_history" in sql_lower and "where id = ?" in sql_lower:
            msg_id = params[0]
            msg = self.messages.get(msg_id)
            if not msg:
                return None
            # Return columns based on SELECT clause
            if "content, metadata" in sql_lower and "id," in sql_lower:
                return (msg["id"], msg["content"], msg["metadata"])
            if "metadata" in sql_lower:
                return (msg["id"], msg["metadata"])
            return (msg["id"], msg["content"], msg["metadata"])

        if "from memory_pins" in sql_lower and "released_at is null" in sql_lower:
            message_id = params[0]
            for pin in self.pins.values():
                if pin["message_id"] == message_id and pin["released_at"] is None:
                    return (pin["id"],)
            return None

        return None

    async def fetchall(self, sql, params=()):
        """Handle SELECT queries returning multiple rows."""
        sql_lower = sql.strip().lower()

        if "from memory_pins" in sql_lower and "join conversation_history" in sql_lower:
            results = []
            for pin in self.pins.values():
                if pin["released_at"] is not None:
                    continue
                msg = self.messages.get(pin["message_id"])
                if msg:
                    results.append((
                        pin["id"],
                        pin["message_id"],
                        pin["pin_reason"],
                        pin["pinned_at"],
                        msg["content"],
                    ))
            return results

        # SELECT pinned_at FROM memory_pins WHERE released_at IS NULL
        if (
            "select pinned_at from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            return [
                (p["pinned_at"],)
                for p in self.pins.values()
                if p["released_at"] is None
            ]

        return []

    async def fetchval(self, sql, params=()):
        """Handle SELECT COUNT(*) and aggregate queries."""
        sql_lower = sql.strip().lower()

        if "count(*)" in sql_lower and "conversation_history" in sql_lower:
            agent_id = params[0] if params else None
            return sum(
                1 for m in self.messages.values()
                if agent_id is None or m["agent_id"] == agent_id
            )

        if "count(*)" in sql_lower and "memory_pins" in sql_lower:
            if "released_at is null" in sql_lower:
                return sum(1 for p in self.pins.values() if p["released_at"] is None)
            if "released_at is not null" in sql_lower:
                return sum(1 for p in self.pins.values() if p["released_at"] is not None)
            return len(self.pins)

        if "min(pinned_at)" in sql_lower:
            active = [p["pinned_at"] for p in self.pins.values() if p["released_at"] is None]
            return min(active) if active else None

        if "max(pinned_at)" in sql_lower:
            active = [p["pinned_at"] for p in self.pins.values() if p["released_at"] is None]
            return max(active) if active else None

        return 0


class FakeGraphStore:
    """In-memory fake graph store for testing KG writes."""

    def __init__(self):
        self.nodes = {}   # node_id -> GraphNode
        self.edges = []   # list of (source_id, target_id, label)

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def add_edge(self, source_id, target_id, label, properties=None):
        self.edges.append((source_id, target_id, label))


def _make_feature(fake_db, agent_id="test-agent", graph_store=None):
    """Create a MemoryAgencyFeature with a mocked agent and fake database."""
    from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature, PIN_QUOTA_DEFAULT

    storage = MagicMock()
    storage.db = fake_db
    storage.agent_id = agent_id
    storage.graph = graph_store

    agent = MagicMock()
    agent.storage = storage

    feature = MemoryAgencyFeature(agent)
    feature.storage = storage
    feature.agent_id = agent_id
    feature.pin_quota = PIN_QUOTA_DEFAULT
    return feature


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pin_memory_sets_decay_protected():
    """Pinning a message should set decay_protected=True in its metadata."""
    db = FakeDB()
    msg_id = db.add_message("Remember this important event", {"importance": 0.8})
    feature = _make_feature(db)

    result = await feature.memory_pin(message_id=msg_id, reason="milestone")

    assert result["pinned"] is True
    assert result["message_id"] == msg_id
    assert "Remember this" in result["preview"]

    # Verify metadata was updated
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True


@pytest.mark.asyncio
async def test_release_memory_clears_pin():
    """Releasing a pinned message should clear decay_protected and set released_at."""
    db = FakeDB()
    msg_id = db.add_message("Temporary note", {"importance": 0.5})
    feature = _make_feature(db)

    # Pin first
    await feature.memory_pin(message_id=msg_id, reason="temp")
    # Release
    result = await feature.memory_release(message_id=msg_id)

    assert result["released"] is True
    assert result["message_id"] == msg_id

    # Metadata should have decay_protected=False
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is False

    # Pin record should have released_at set
    active_pins = [p for p in db.pins.values() if p["released_at"] is None]
    assert len(active_pins) == 0


@pytest.mark.asyncio
async def test_list_pinned_returns_active_pins():
    """memory_pinned should return only non-released pins."""
    db = FakeDB()
    msg1 = db.add_message("First memory")
    msg2 = db.add_message("Second memory")
    msg3 = db.add_message("Third memory")
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg1, reason="reason A")
    await feature.memory_pin(message_id=msg2, reason="reason B")
    await feature.memory_pin(message_id=msg3, reason="reason C")

    # Release the second one
    await feature.memory_release(message_id=msg2)

    result = await feature.memory_pinned()

    assert result["count"] == 2
    pinned_ids = {p["message_id"] for p in result["pins"]}
    assert msg1 in pinned_ids
    assert msg3 in pinned_ids
    assert msg2 not in pinned_ids


@pytest.mark.asyncio
async def test_pin_stats_returns_ratios():
    """memory_pin_stats should return correct counts and ratios."""
    db = FakeDB()
    # Add 10 messages
    ids = [db.add_message(f"Message {i}") for i in range(10)]
    feature = _make_feature(db)

    # Pin 3 messages
    await feature.memory_pin(message_id=ids[0])
    await feature.memory_pin(message_id=ids[1])
    await feature.memory_pin(message_id=ids[2])

    # Release 1
    await feature.memory_release(message_id=ids[1])

    result = await feature.memory_pin_stats()

    assert result["total_messages"] == 10
    assert result["pinned"] == 2       # 3 pinned - 1 released = 2 active
    assert result["released"] == 1
    assert result["pin_ratio"] == 0.2  # 2 / 10


@pytest.mark.asyncio
async def test_double_pin_is_idempotent():
    """Pinning the same message twice should not create a duplicate pin record."""
    db = FakeDB()
    msg_id = db.add_message("Pin me twice")
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg_id, reason="first pin")
    await feature.memory_pin(message_id=msg_id, reason="second pin")

    # Should still have only one active pin
    active_pins = [p for p in db.pins.values() if p["released_at"] is None and p["message_id"] == msg_id]
    assert len(active_pins) == 1

    # Metadata should still be protected
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True


@pytest.mark.asyncio
async def test_pin_nonexistent_message_returns_error():
    """Pinning a message that does not exist should return an error."""
    db = FakeDB()
    feature = _make_feature(db)

    result = await feature.memory_pin(message_id=99999)

    assert "error" in result
    assert "99999" in result["error"]


@pytest.mark.asyncio
async def test_release_nonexistent_message_returns_error():
    """Releasing a message that does not exist should return an error."""
    db = FakeDB()
    feature = _make_feature(db)

    result = await feature.memory_release(message_id=99999)

    assert "error" in result
    assert "99999" in result["error"]


@pytest.mark.asyncio
async def test_pinned_memory_boost_in_retriever():
    """Verify that _calculate_score boosts importance for decay_protected memories."""
    from kestrel_sovereign.storage.memory_retriever import MemoryRetriever

    store = AsyncMock()
    retriever = MemoryRetriever(conversation_store=store)

    # Score with decay_protected = True and low base importance
    score_pinned = retriever._calculate_score(
        content="test content",
        query="test",
        metadata={"importance": 0.3, "decay_protected": True},
        emotional_context=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        expanded_concepts=[],
    )

    # Score with decay_protected = False and same low base importance
    score_unpinned = retriever._calculate_score(
        content="test content",
        query="test",
        metadata={"importance": 0.3, "decay_protected": False},
        emotional_context=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        expanded_concepts=[],
    )

    # Pinned should score higher due to importance boost
    assert score_pinned > score_unpinned


@pytest.mark.asyncio
async def test_pin_preserves_existing_metadata():
    """Pinning should preserve existing metadata fields while adding decay_protected."""
    db = FakeDB()
    original_meta = {
        "importance": 0.8,
        "emotional_valence": 0.6,
        "emotional_categories": ["joy"],
        "custom_field": "preserved",
    }
    msg_id = db.add_message("Important joyful memory", original_meta)
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg_id, reason="important")

    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True
    assert stored_meta["importance"] == 0.8
    assert stored_meta["emotional_valence"] == 0.6
    assert stored_meta["emotional_categories"] == ["joy"]
    assert stored_meta["custom_field"] == "preserved"


# --------------------------------------------------------------------------
# save_fact tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_fact_creates_kg_node():
    """save_fact should create a learned_fact node in the knowledge graph."""
    db = FakeDB()
    graph = FakeGraphStore()
    feature = _make_feature(db, graph_store=graph)

    result = await feature.save_fact(
        subject="user", predicate="favorite_number", value="445"
    )

    assert result["saved"] is True
    assert result["subject"] == "user"
    assert result["predicate"] == "favorite_number"
    assert result["value"] == "445"

    # Verify KG node was created
    fact_id = result["node_id"]
    assert fact_id in graph.nodes
    node = graph.nodes[fact_id]
    assert node.node_type == "learned_fact"
    assert node.label == "Favorite Number: 445"
    assert node.properties["subject"] == "user"
    assert node.properties["predicate"] == "favorite_number"
    assert node.properties["value"] == "445"
    assert node.properties["confidence"] == 1.0
    assert node.properties["source"] == "agent_tool"

    # Verify edge was created
    assert ("test-agent", fact_id, "knows") in graph.edges


@pytest.mark.asyncio
async def test_save_fact_upserts_same_subject_predicate():
    """Saving the same subject+predicate should update the existing node."""
    db = FakeDB()
    graph = FakeGraphStore()
    feature = _make_feature(db, graph_store=graph)

    await feature.save_fact(subject="user", predicate="favorite_color", value="blue")
    result = await feature.save_fact(subject="user", predicate="favorite_color", value="green")

    assert result["saved"] is True
    assert result["value"] == "green"

    # Should still be one node (upserted)
    fact_id = "fact:test-agent:user:favorite_color"
    assert graph.nodes[fact_id].label == "Favorite Color: green"
    assert graph.nodes[fact_id].properties["value"] == "green"


@pytest.mark.asyncio
async def test_save_fact_clamps_confidence():
    """Confidence should be clamped to [0.0, 1.0]."""
    db = FakeDB()
    graph = FakeGraphStore()
    feature = _make_feature(db, graph_store=graph)

    result = await feature.save_fact(
        subject="user", predicate="test", value="x", confidence=2.5
    )
    assert result["confidence"] == 1.0

    result = await feature.save_fact(
        subject="user", predicate="test2", value="y", confidence=-0.5
    )
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_save_fact_without_graph_returns_error():
    """save_fact should return error if graph store is not available."""
    db = FakeDB()
    feature = _make_feature(db, graph_store=None)

    result = await feature.save_fact(subject="user", predicate="name", value="Alice")

    assert "error" in result
    assert "not available" in result["error"]
