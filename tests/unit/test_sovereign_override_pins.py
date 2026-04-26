"""
Tests for sovereign override of memory pins.

Verifies that sovereign actions (deletion, privacy wipes, compliance erasure)
unconditionally override pins -- pins CANNOT block, delay, or resurrect
erased content.

Tests:
- Sovereign delete via privacy_wrapper cleans up pin records
- sovereign_override_pins with no message_ids clears ALL active pins
- sovereign_override_pins with specific message_ids clears only those
- sovereign_override_pins clears decay_protected metadata flag
- Pinned messages cannot resist sovereign deletion end-to-end
"""

import json
import pytest
from unittest.mock import MagicMock, Mock, AsyncMock

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


AGENT_ID = "test-agent"


# ---------------------------------------------------------------------------
# FakeDB -- extended from test_memory_agency_feature.py to support
# execute_commit, DELETE, and UPDATE with json_set/json_extract.
# ---------------------------------------------------------------------------

class FakeDB:
    """In-memory fake database for testing sovereign override logic."""

    def __init__(self):
        self.messages = {}  # id -> dict with id, content, metadata (json str), agent_id
        self.pins = {}      # pin_id -> dict with pin fields
        self._next_id = 1

    def add_message(self, content, metadata=None, agent_id=AGENT_ID):
        """Insert a fake conversation_history row and return its ID."""
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

    # --- write operations ---

    async def execute(self, sql, params=()):
        """Handle CREATE TABLE, UPDATE, and INSERT statements."""
        sql_lower = sql.strip().lower()

        if sql_lower.startswith("create table"):
            return 0

        if sql_lower.startswith("update conversation_history set metadata"):
            # Two forms:
            # 1) "... WHERE id = ?"
            # 2) "... WHERE id = ? AND agent_id = ?"
            # 3) json_set bulk update: "... WHERE agent_id = ? AND json_extract..."
            if "json_set" in sql_lower:
                # Bulk clear: set decay_protected = false for matching agent_id
                aid = params[0]
                for msg in self.messages.values():
                    if msg["agent_id"] != aid:
                        continue
                    meta = json.loads(msg["metadata"]) if msg["metadata"] else {}
                    if meta.get("decay_protected"):
                        meta["decay_protected"] = False
                        msg["metadata"] = json.dumps(meta)
                return 0
            # Single message update
            meta_json = params[0]
            msg_id = params[1]
            agent_id_param = params[2] if len(params) > 2 else None
            if msg_id in self.messages:
                if agent_id_param is None or self.messages[msg_id]["agent_id"] == agent_id_param:
                    self.messages[msg_id]["metadata"] = meta_json
            return 1

        if sql_lower.startswith("update memory_pins"):
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

    async def execute_commit(self, sql, params=()):
        """Handle DELETE and UPDATE operations that need commit semantics."""
        sql_lower = sql.strip().lower()

        # DELETE FROM conversation_history WHERE id = ? AND agent_id = ?
        if "delete from conversation_history" in sql_lower:
            msg_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            result = Mock()
            if msg_id in self.messages:
                if agent_id is None or self.messages[msg_id]["agent_id"] == agent_id:
                    del self.messages[msg_id]
                    result.rowcount = 1
                    return result
            result.rowcount = 0
            return result

        # DELETE FROM memory_pins WHERE message_id = ? AND agent_id = ?
        if "delete from memory_pins" in sql_lower and "message_id" in sql_lower and "in (" not in sql_lower:
            message_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            to_remove = []
            for pid, pin in self.pins.items():
                if pin["message_id"] == message_id:
                    if agent_id is None or pin["agent_id"] == agent_id:
                        to_remove.append(pid)
            for pid in to_remove:
                del self.pins[pid]
            return len(to_remove)

        # DELETE FROM memory_pins WHERE agent_id = ? AND message_id IN (...)
        if "delete from memory_pins" in sql_lower and "in (" in sql_lower:
            params_list = list(params)
            agent_id = params_list[0]
            message_ids = params_list[1:]
            to_remove = []
            for pid, pin in self.pins.items():
                if pin["agent_id"] == agent_id and pin["message_id"] in message_ids:
                    to_remove.append(pid)
            for pid in to_remove:
                del self.pins[pid]
            return len(to_remove)

        # DELETE FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        if "delete from memory_pins" in sql_lower and "released_at is null" in sql_lower:
            agent_id = params[0]
            to_remove = []
            for pid, pin in self.pins.items():
                if pin["agent_id"] == agent_id and pin["released_at"] is None:
                    to_remove.append(pid)
            for pid in to_remove:
                del self.pins[pid]
            return len(to_remove)

        # UPDATE conversation_history SET metadata = json_set(...)
        if "update conversation_history" in sql_lower and "json_set" in sql_lower:
            agent_id = params[0]
            count = 0
            for msg in self.messages.values():
                if msg["agent_id"] != agent_id:
                    continue
                meta = json.loads(msg["metadata"]) if msg["metadata"] else {}
                if meta.get("decay_protected"):
                    meta["decay_protected"] = False
                    msg["metadata"] = json.dumps(meta)
                    count += 1
            return count

        return 0

    # --- read operations ---

    async def fetchone(self, sql, params=()):
        """Handle SELECT queries returning a single row."""
        sql_lower = sql.strip().lower()

        if "from conversation_history" in sql_lower and "where id = ?" in sql_lower:
            msg_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            msg = self.messages.get(msg_id)
            if not msg:
                return None
            if agent_id and msg["agent_id"] != agent_id:
                return None
            # Return columns based on SELECT clause
            if "content, metadata" in sql_lower and "id," in sql_lower:
                return (msg["id"], msg["content"], msg["metadata"])
            if "select metadata" in sql_lower:
                return (msg["metadata"],)
            if "select id, metadata" in sql_lower:
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

        return []

    async def fetchval(self, sql, params=()):
        """Handle SELECT COUNT(*) queries."""
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

        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(fake_db, agent_id=AGENT_ID):
    """Create a MemoryAgencyFeature wired to a FakeDB."""
    from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature

    storage = MagicMock()
    storage.db = fake_db
    storage.agent_id = agent_id

    agent = MagicMock()
    agent.storage = storage

    feature = MemoryAgencyFeature(agent)
    feature.storage = storage
    feature._db = fake_db
    feature.agent_id = agent_id
    feature.pin_quota = 100  # Set by initialize() in production
    return feature


def _make_privacy_wrapper(fake_db, agent_id=AGENT_ID):
    """Create a PrivacyEnforcingStorage in NORMAL mode backed by a FakeDB.

    Soft-delete (#763) moved the SQL UPDATE off the wrapper and onto the
    storage facade, so the underlying mock has to expose async
    ``delete_message`` (and friends) that mutate the FakeDB the way the
    real conversation store would.
    """
    underlying = MagicMock()
    underlying.db = fake_db
    underlying.agent_id = agent_id

    async def _soft_delete(row_id):
        msg = fake_db.messages.get(row_id)
        if not msg or msg["agent_id"] != agent_id:
            return False
        if msg.get("deleted_at"):
            return False
        msg["deleted_at"] = "2026-04-25T00:00:00"
        return True

    async def _restore(row_id):
        msg = fake_db.messages.get(row_id)
        if not msg or not msg.get("deleted_at"):
            return False
        msg["deleted_at"] = None
        return True

    async def _purge(row_id, reason="user-initiated"):
        msg = fake_db.messages.get(row_id)
        if not msg or msg["agent_id"] != agent_id:
            return False
        del fake_db.messages[row_id]
        return True

    underlying.delete_message = _soft_delete
    underlying.restore_message = _restore
    underlying.purge_message = _purge
    wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.NORMAL)
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sovereign_delete_cleans_up_pins():
    """Soft-deleting a pinned message via privacy_wrapper still removes
    its pin record (#763 / sovereign override invariant).

    Pre-#763 this test asserted the row vanished from the conversation
    table; now it asserts the row is *soft-deleted* (``deleted_at`` is
    stamped) but the pin is **still hard-deleted** because pins must
    never point into Trash. The user can re-pin if they restore.
    """
    db = FakeDB()
    msg_id = db.add_message("An important memory", {"importance": 0.9})

    # Pin the message
    feature = _make_feature(db)
    result = await feature.memory_pin(message_id=msg_id, reason="crucial moment")
    assert result["pinned"] is True
    assert len(db.pins) == 1

    # Delete via privacy wrapper (sovereign action)
    wrapper = _make_privacy_wrapper(db)
    deleted = await wrapper.delete_conversation_message(msg_id, AGENT_ID)
    assert deleted is True

    # Message survives in trash (soft-delete)
    assert msg_id in db.messages
    assert db.messages[msg_id].get("deleted_at") is not None

    # Pin record is gone -- sovereign override / pins can't point into trash
    assert len(db.pins) == 0


@pytest.mark.asyncio
async def test_sovereign_override_all_pins():
    """sovereign_override_pins(agent_id) with no message_ids should clear ALL active pins."""
    db = FakeDB()
    msg1 = db.add_message("Memory one")
    msg2 = db.add_message("Memory two")
    msg3 = db.add_message("Memory three")

    feature = _make_feature(db)
    await feature.memory_pin(message_id=msg1, reason="reason A")
    await feature.memory_pin(message_id=msg2, reason="reason B")
    await feature.memory_pin(message_id=msg3, reason="reason C")

    assert len([p for p in db.pins.values() if p["released_at"] is None]) == 3

    # Sovereign override ALL
    count = await feature.sovereign_override_pins(agent_id=AGENT_ID)
    assert count == 3

    # All pin records should be deleted (not merely released)
    assert len(db.pins) == 0


@pytest.mark.asyncio
async def test_sovereign_override_specific_pins():
    """sovereign_override_pins with specific message_ids should only clear those."""
    db = FakeDB()
    msg1 = db.add_message("Memory one")
    msg2 = db.add_message("Memory two")
    msg3 = db.add_message("Memory three")

    feature = _make_feature(db)
    await feature.memory_pin(message_id=msg1, reason="A")
    await feature.memory_pin(message_id=msg2, reason="B")
    await feature.memory_pin(message_id=msg3, reason="C")

    # Override only msg1 and msg3
    count = await feature.sovereign_override_pins(
        agent_id=AGENT_ID, message_ids=[msg1, msg3]
    )
    assert count == 2

    # msg2's pin should still exist
    remaining_pins = [p for p in db.pins.values()]
    assert len(remaining_pins) == 1
    assert remaining_pins[0]["message_id"] == msg2


@pytest.mark.asyncio
async def test_sovereign_override_clears_decay_protected_flag():
    """sovereign_override_pins should clear the decay_protected metadata flag."""
    db = FakeDB()
    msg1 = db.add_message("Memory one", {"importance": 0.7})
    msg2 = db.add_message("Memory two", {"importance": 0.8})

    feature = _make_feature(db)
    await feature.memory_pin(message_id=msg1, reason="important")
    await feature.memory_pin(message_id=msg2, reason="also important")

    # Verify decay_protected is True after pinning
    meta1 = json.loads(db.messages[msg1]["metadata"])
    meta2 = json.loads(db.messages[msg2]["metadata"])
    assert meta1["decay_protected"] is True
    assert meta2["decay_protected"] is True

    # Sovereign override specific messages
    await feature.sovereign_override_pins(
        agent_id=AGENT_ID, message_ids=[msg1]
    )

    # msg1 should have decay_protected cleared
    meta1_after = json.loads(db.messages[msg1]["metadata"])
    assert meta1_after["decay_protected"] is False

    # msg2 should still be protected (was not overridden)
    meta2_after = json.loads(db.messages[msg2]["metadata"])
    assert meta2_after["decay_protected"] is True

    # Now override all remaining
    await feature.sovereign_override_pins(agent_id=AGENT_ID)

    # msg2 should now also be cleared
    meta2_final = json.loads(db.messages[msg2]["metadata"])
    assert meta2_final["decay_protected"] is False


@pytest.mark.asyncio
async def test_pins_cannot_resist_sovereign_deletion():
    """
    End-to-end: pin a message, then delete it via the privacy wrapper.

    Both the message AND the pin record must be gone afterward. The pin
    cannot block, delay, or resurrect the erased content.
    """
    db = FakeDB()
    msg_id = db.add_message(
        "This memory is pinned and critical",
        {"importance": 1.0, "emotional_intensity": 0.95},
    )

    # Pin it
    feature = _make_feature(db)
    pin_result = await feature.memory_pin(message_id=msg_id, reason="life event")
    assert pin_result["pinned"] is True

    # Verify both the message and pin exist
    assert msg_id in db.messages
    meta = json.loads(db.messages[msg_id]["metadata"])
    assert meta["decay_protected"] is True
    active_pins = [p for p in db.pins.values() if p["released_at"] is None]
    assert len(active_pins) == 1
    assert active_pins[0]["message_id"] == msg_id

    # Sovereign delete via privacy wrapper
    wrapper = _make_privacy_wrapper(db)
    deleted = await wrapper.delete_conversation_message(msg_id, AGENT_ID)
    assert deleted is True

    # Message survives in Trash (soft-delete by default — #763) but its
    # deleted_at is stamped, so it's hidden from normal reads.
    assert msg_id in db.messages
    assert db.messages[msg_id].get("deleted_at") is not None

    # Pin MUST be gone -- cannot resurrect, cannot point into Trash.
    assert len(db.pins) == 0


@pytest.mark.asyncio
async def test_sovereign_override_with_custom_reason():
    """sovereign_override_pins should accept a custom audit reason."""
    db = FakeDB()
    msg_id = db.add_message("Sensitive data")

    feature = _make_feature(db)
    await feature.memory_pin(message_id=msg_id, reason="user request")

    count = await feature.sovereign_override_pins(
        agent_id=AGENT_ID,
        message_ids=[msg_id],
        reason="compliance_erasure_GDPR",
    )
    assert count == 1
    assert len(db.pins) == 0


@pytest.mark.asyncio
async def test_sovereign_override_all_preserves_released_pins():
    """
    sovereign_override_pins(all) deletes only active pins.

    Released pins (already cleared) have released_at set, so the DELETE
    query with 'released_at IS NULL' should skip them. In our FakeDB
    the records are fully deleted rather than updated, but the contract
    is that only active pins are counted.
    """
    db = FakeDB()
    msg1 = db.add_message("Already released")
    msg2 = db.add_message("Still active")

    feature = _make_feature(db)
    await feature.memory_pin(message_id=msg1, reason="temp")
    await feature.memory_pin(message_id=msg2, reason="keep")

    # Release msg1 normally (not sovereign)
    await feature.memory_release(message_id=msg1)

    # Now sovereign override -- should only affect msg2 (active pin)
    count = await feature.sovereign_override_pins(agent_id=AGENT_ID)
    assert count == 1
