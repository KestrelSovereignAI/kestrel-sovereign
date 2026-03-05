"""
Tests for pin quota enforcement, monitoring, and admin bulk-unpin.

Verifies:
- Pin quota is enforced at the configured limit
- Pin quota is configurable per-instance
- Pin ratio warning is emitted when threshold is exceeded
- Admin bulk-unpin-all removes all active pins
- Admin unpin-oldest removes only the N oldest pins
- Pin stats include quota, ratio, alert, and age information
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock


class FakeDB:
    """In-memory fake database for testing the memory agency feature.

    Supports the full set of SQL patterns used by MemoryAgencyFeature
    including quota checks, admin bulk-unpin, and enhanced stats queries.
    """

    def __init__(self):
        self.messages = {}  # id -> dict
        self.pins = {}      # id -> dict
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

    # -- write operations --------------------------------------------------

    async def execute(self, sql, params=()):
        sql_lower = sql.strip().lower()

        if sql_lower.startswith("create table"):
            return 0

        # UPDATE conversation_history SET metadata = ? WHERE id = ?
        if sql_lower.startswith("update conversation_history"):
            meta_json, msg_id = params
            if msg_id in self.messages:
                self.messages[msg_id]["metadata"] = meta_json
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE id = ?
        if (
            sql_lower.startswith("update memory_pins")
            and "where id = ?" in sql_lower
        ):
            released_at, pin_id = params
            if pin_id in self.pins:
                self.pins[pin_id]["released_at"] = released_at
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE released_at IS NULL
        # (bulk unpin-all -- no message_id filter)
        if (
            sql_lower.startswith("update memory_pins")
            and "released_at is null" in sql_lower
            and "message_id" not in sql_lower
        ):
            released_at = params[0]
            for pin in self.pins.values():
                if pin["released_at"] is None:
                    pin["released_at"] = released_at
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE message_id = ? AND released_at IS NULL
        if (
            sql_lower.startswith("update memory_pins")
            and "message_id" in sql_lower
        ):
            released_at, message_id = params
            for pin in self.pins.values():
                if pin["message_id"] == message_id and pin["released_at"] is None:
                    pin["released_at"] = released_at
            return 1

        # INSERT INTO memory_pins
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

    # -- single-row reads --------------------------------------------------

    async def fetchone(self, sql, params=()):
        sql_lower = sql.strip().lower()

        # SELECT ... FROM conversation_history WHERE id = ?
        if "from conversation_history" in sql_lower and "where id = ?" in sql_lower:
            msg_id = params[0]
            msg = self.messages.get(msg_id)
            if not msg:
                return None
            if "content, metadata" in sql_lower and "id," in sql_lower:
                return (msg["id"], msg["content"], msg["metadata"])
            if "metadata" in sql_lower:
                return (msg["id"], msg["metadata"])
            return (msg["id"], msg["content"], msg["metadata"])

        # SELECT id FROM memory_pins WHERE message_id = ? AND released_at IS NULL
        if "from memory_pins" in sql_lower and "released_at is null" in sql_lower:
            message_id = params[0]
            for pin in self.pins.values():
                if pin["message_id"] == message_id and pin["released_at"] is None:
                    return (pin["id"],)
            return None

        return None

    # -- multi-row reads ---------------------------------------------------

    async def fetchall(self, sql, params=()):
        sql_lower = sql.strip().lower()

        # JOIN query for memory_pinned()
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

        # SELECT message_id FROM memory_pins WHERE released_at IS NULL
        # (used by admin_unpin_all)
        if (
            "select message_id from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            return [
                (p["message_id"],)
                for p in self.pins.values()
                if p["released_at"] is None
            ]

        # SELECT id, message_id FROM memory_pins WHERE released_at IS NULL
        # ORDER BY pinned_at ASC LIMIT ?
        # (used by admin_unpin_oldest)
        if (
            "select id, message_id from memory_pins" in sql_lower
            and "order by pinned_at asc" in sql_lower
        ):
            limit = params[0] if params else None
            active = sorted(
                [p for p in self.pins.values() if p["released_at"] is None],
                key=lambda p: p["pinned_at"],
            )
            if limit is not None:
                active = active[:limit]
            return [(p["id"], p["message_id"]) for p in active]

        # SELECT pinned_at FROM memory_pins WHERE released_at IS NULL
        # (used by stats average pin age)
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

    # -- scalar reads ------------------------------------------------------

    async def fetchval(self, sql, params=()):
        sql_lower = sql.strip().lower()

        # COUNT(*) on conversation_history
        if "count(*)" in sql_lower and "conversation_history" in sql_lower:
            agent_id = params[0] if params else None
            return sum(
                1 for m in self.messages.values()
                if agent_id is None or m["agent_id"] == agent_id
            )

        # COUNT(*) on memory_pins
        if "count(*)" in sql_lower and "memory_pins" in sql_lower:
            if "released_at is null" in sql_lower:
                return sum(1 for p in self.pins.values() if p["released_at"] is None)
            if "released_at is not null" in sql_lower:
                return sum(1 for p in self.pins.values() if p["released_at"] is not None)
            return len(self.pins)

        # MIN(pinned_at) FROM memory_pins WHERE released_at IS NULL
        if "min(pinned_at)" in sql_lower:
            active = [p["pinned_at"] for p in self.pins.values() if p["released_at"] is None]
            return min(active) if active else None

        # MAX(pinned_at) FROM memory_pins WHERE released_at IS NULL
        if "max(pinned_at)" in sql_lower:
            active = [p["pinned_at"] for p in self.pins.values() if p["released_at"] is None]
            return max(active) if active else None

        return 0


def _make_feature(fake_db, agent_id="test-agent", pin_quota=None):
    """Create a MemoryAgencyFeature with a mocked agent and fake database."""
    from kestrel_sovereign.features.memory_agency.feature import (
        MemoryAgencyFeature,
        PIN_QUOTA_DEFAULT,
    )

    storage = MagicMock()
    storage.db = fake_db
    storage.agent_id = agent_id

    agent = MagicMock()
    agent.storage = storage

    feature = MemoryAgencyFeature(agent)
    feature.storage = storage
    feature.agent_id = agent_id
    feature.pin_quota = pin_quota if pin_quota is not None else PIN_QUOTA_DEFAULT
    return feature


# --------------------------------------------------------------------------
# Quota enforcement
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pin_quota_enforced():
    """Pinning beyond the default quota (100) should be rejected."""
    from kestrel_sovereign.features.memory_agency.feature import PIN_QUOTA_DEFAULT

    db = FakeDB()
    feature = _make_feature(db)

    # Create 101 messages
    msg_ids = [db.add_message(f"Message {i}") for i in range(PIN_QUOTA_DEFAULT + 1)]

    # Pin up to the quota limit -- all should succeed
    for i in range(PIN_QUOTA_DEFAULT):
        result = await feature.memory_pin(message_id=msg_ids[i], reason=f"pin {i}")
        assert result["pinned"] is True, f"Pin {i} should succeed"

    # The 101st pin should be rejected
    result = await feature.memory_pin(
        message_id=msg_ids[PIN_QUOTA_DEFAULT], reason="one too many"
    )
    assert result.get("pinned") is False
    assert "error" in result
    assert "quota" in result["error"].lower()


@pytest.mark.asyncio
async def test_pin_quota_configurable():
    """Setting a custom quota should be enforced at that limit."""
    db = FakeDB()
    custom_quota = 5
    feature = _make_feature(db, pin_quota=custom_quota)

    msg_ids = [db.add_message(f"Message {i}") for i in range(custom_quota + 1)]

    for i in range(custom_quota):
        result = await feature.memory_pin(message_id=msg_ids[i])
        assert result["pinned"] is True

    # The (custom_quota + 1)th pin should fail
    result = await feature.memory_pin(message_id=msg_ids[custom_quota])
    assert result.get("pinned") is False
    assert "error" in result
    assert str(custom_quota) in result["error"]


@pytest.mark.asyncio
async def test_pin_quota_repin_does_not_double_count():
    """Re-pinning an already-pinned message should not count against quota."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=2)

    msg_ids = [db.add_message(f"Message {i}") for i in range(3)]

    # Pin two messages (fills quota)
    await feature.memory_pin(message_id=msg_ids[0])
    await feature.memory_pin(message_id=msg_ids[1])

    # Re-pin the first one -- should succeed (idempotent, not a new pin)
    result = await feature.memory_pin(message_id=msg_ids[0], reason="re-pin")
    assert result["pinned"] is True

    # A genuinely new pin should still be rejected
    result = await feature.memory_pin(message_id=msg_ids[2])
    assert result.get("pinned") is False


# --------------------------------------------------------------------------
# Ratio warning
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pin_ratio_warning():
    """When pin ratio exceeds the threshold, pin response includes a warning."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create 4 messages, pin 3 -> ratio = 3/4 = 0.75 > 0.5 threshold
    msg_ids = [db.add_message(f"Message {i}") for i in range(4)]

    await feature.memory_pin(message_id=msg_ids[0])
    await feature.memory_pin(message_id=msg_ids[1])

    # Pin the third -- this pushes ratio to 3/4 = 75%
    result = await feature.memory_pin(message_id=msg_ids[2])

    assert result["pinned"] is True
    assert "warning" in result
    assert "ratio" in result["warning"].lower()


@pytest.mark.asyncio
async def test_pin_ratio_no_warning_below_threshold():
    """When pin ratio is at or below the threshold, no warning is emitted."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create 10 messages, pin 3 -> ratio = 3/10 = 0.3 < 0.5
    msg_ids = [db.add_message(f"Message {i}") for i in range(10)]

    await feature.memory_pin(message_id=msg_ids[0])
    await feature.memory_pin(message_id=msg_ids[1])
    result = await feature.memory_pin(message_id=msg_ids[2])

    assert result["pinned"] is True
    assert "warning" not in result


# --------------------------------------------------------------------------
# Admin bulk-unpin
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_bulk_unpin_all():
    """admin_unpin_all should release every active pin and clear metadata."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    msg_ids = [db.add_message(f"Message {i}") for i in range(5)]
    for mid in msg_ids:
        await feature.memory_pin(message_id=mid, reason="keep")

    # Verify 5 active pins before bulk unpin
    active_before = sum(1 for p in db.pins.values() if p["released_at"] is None)
    assert active_before == 5

    result = await feature.memory_admin_unpin_all()

    assert result["unpinned"] == 5

    # All pins should now be released
    active_after = sum(1 for p in db.pins.values() if p["released_at"] is None)
    assert active_after == 0

    # All messages should have decay_protected = False
    for mid in msg_ids:
        meta = json.loads(db.messages[mid]["metadata"])
        assert meta["decay_protected"] is False


@pytest.mark.asyncio
async def test_admin_bulk_unpin_all_empty():
    """admin_unpin_all with no active pins should return unpinned=0."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    result = await feature.memory_admin_unpin_all()
    assert result["unpinned"] == 0


# --------------------------------------------------------------------------
# Admin unpin-oldest
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_unpin_oldest():
    """admin_unpin_oldest should remove only the N oldest pins."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create 5 messages and pin them with incrementing timestamps
    msg_ids = [db.add_message(f"Message {i}") for i in range(5)]
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for i, mid in enumerate(msg_ids):
        await feature.memory_pin(message_id=mid, reason=f"pin {i}")
        # Override pinned_at for deterministic ordering
        for pin in db.pins.values():
            if pin["message_id"] == mid and pin["released_at"] is None:
                pin["pinned_at"] = (base + timedelta(hours=i)).isoformat()

    # Unpin the 2 oldest
    result = await feature.memory_admin_unpin_oldest(count=2)

    assert result["unpinned"] == 2
    assert result["requested"] == 2

    # The 2 oldest messages should be unpinned; the 3 newest should remain
    active_message_ids = {
        p["message_id"] for p in db.pins.values() if p["released_at"] is None
    }
    assert len(active_message_ids) == 3
    # msg_ids[0] and msg_ids[1] were the oldest and should be gone
    assert msg_ids[0] not in active_message_ids
    assert msg_ids[1] not in active_message_ids
    # msg_ids[2..4] should still be active
    assert msg_ids[2] in active_message_ids
    assert msg_ids[3] in active_message_ids
    assert msg_ids[4] in active_message_ids

    # Verify metadata was cleared on the unpinned messages
    for mid in [msg_ids[0], msg_ids[1]]:
        meta = json.loads(db.messages[mid]["metadata"])
        assert meta["decay_protected"] is False

    # Verify metadata still protected on remaining pins
    for mid in [msg_ids[2], msg_ids[3], msg_ids[4]]:
        meta = json.loads(db.messages[mid]["metadata"])
        assert meta["decay_protected"] is True


@pytest.mark.asyncio
async def test_admin_unpin_oldest_more_than_exist():
    """Requesting to unpin more than the number of active pins unpins all."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    msg_ids = [db.add_message(f"Message {i}") for i in range(3)]
    for mid in msg_ids:
        await feature.memory_pin(message_id=mid)

    result = await feature.memory_admin_unpin_oldest(count=10)

    assert result["unpinned"] == 3
    assert result["requested"] == 10

    active_after = sum(1 for p in db.pins.values() if p["released_at"] is None)
    assert active_after == 0


# --------------------------------------------------------------------------
# Enhanced pin stats
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pin_stats_includes_quota_info():
    """Pin stats should include quota, quota_remaining, and age info."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=50)

    msg_ids = [db.add_message(f"Message {i}") for i in range(10)]

    # Pin 3 messages
    await feature.memory_pin(message_id=msg_ids[0])
    await feature.memory_pin(message_id=msg_ids[1])
    await feature.memory_pin(message_id=msg_ids[2])

    stats = await feature.memory_pin_stats()

    assert stats["total_messages"] == 10
    assert stats["pinned"] == 3
    assert stats["quota"] == 50
    assert stats["quota_remaining"] == 47
    assert stats["pin_ratio"] == 0.3
    # Age fields should be present (non-None since we have pins)
    assert stats["oldest_pin_age_seconds"] is not None
    assert stats["average_pin_age_seconds"] is not None
    # No alert at 30% ratio
    assert "alert" not in stats


@pytest.mark.asyncio
async def test_pin_stats_alert_when_ratio_exceeds_threshold():
    """Pin stats should include an alert when ratio > 50%."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create 4 messages, pin 3 -> ratio = 3/4 = 0.75
    msg_ids = [db.add_message(f"Message {i}") for i in range(4)]
    await feature.memory_pin(message_id=msg_ids[0])
    await feature.memory_pin(message_id=msg_ids[1])
    await feature.memory_pin(message_id=msg_ids[2])

    stats = await feature.memory_pin_stats()

    assert stats["pin_ratio"] == 0.75
    assert "alert" in stats
    assert "threshold" in stats["alert"].lower()


@pytest.mark.asyncio
async def test_pin_stats_no_pins():
    """Pin stats with no pins should have zeroes and no alert."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create some messages but pin none
    for i in range(5):
        db.add_message(f"Message {i}")

    stats = await feature.memory_pin_stats()

    assert stats["pinned"] == 0
    assert stats["quota"] == 100
    assert stats["quota_remaining"] == 100
    assert stats["pin_ratio"] == 0.0
    assert stats["oldest_pin_age_seconds"] is None
    assert stats["average_pin_age_seconds"] is None
    assert "alert" not in stats


@pytest.mark.asyncio
async def test_pin_stats_quota_remaining_floor_at_zero():
    """quota_remaining should never go below zero (e.g. if pins were added
    before a quota reduction)."""
    db = FakeDB()
    # Start with high quota, pin many, then lower the quota
    feature = _make_feature(db, pin_quota=100)
    msg_ids = [db.add_message(f"Message {i}") for i in range(10)]
    for mid in msg_ids:
        await feature.memory_pin(message_id=mid)

    # Now reduce quota below current pin count
    feature.pin_quota = 5

    stats = await feature.memory_pin_stats()
    assert stats["pinned"] == 10
    assert stats["quota"] == 5
    assert stats["quota_remaining"] == 0  # floor at zero, not negative
