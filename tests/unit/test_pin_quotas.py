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

from kestrel_sdk.tools.result import ToolResultStatus


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

        # UPDATE conversation_history SET metadata = ? WHERE id = ? AND agent_id = ? AND deleted_at IS NULL
        if sql_lower.startswith("update conversation_history"):
            meta_json, msg_id = params[0], params[1]
            agent_id = params[2] if len(params) > 2 else None
            msg = self.messages.get(msg_id)
            if msg and (agent_id is None or msg["agent_id"] == agent_id):
                msg["metadata"] = meta_json
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE id = ? AND agent_id = ?
        if (
            sql_lower.startswith("update memory_pins")
            and "where id = ?" in sql_lower
        ):
            released_at, pin_id = params[0], params[1]
            agent_id = params[2] if len(params) > 2 else None
            pin = self.pins.get(pin_id)
            if pin and (agent_id is None or pin["agent_id"] == agent_id):
                pin["released_at"] = released_at
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE agent_id = ? AND released_at IS NULL
        # (bulk unpin-all -- no message_id filter)
        if (
            sql_lower.startswith("update memory_pins")
            and "released_at is null" in sql_lower
            and "message_id" not in sql_lower
        ):
            released_at = params[0]
            agent_id = params[1] if len(params) > 1 else None
            for pin in self.pins.values():
                if pin["released_at"] is None and (
                    agent_id is None or pin["agent_id"] == agent_id
                ):
                    pin["released_at"] = released_at
            return 1

        # UPDATE memory_pins SET released_at = ? WHERE message_id = ? AND agent_id = ? AND released_at IS NULL
        if (
            sql_lower.startswith("update memory_pins")
            and "message_id" in sql_lower
        ):
            released_at, message_id = params[0], params[1]
            agent_id = params[2] if len(params) > 2 else None
            for pin in self.pins.values():
                if (
                    pin["message_id"] == message_id
                    and pin["released_at"] is None
                    and (agent_id is None or pin["agent_id"] == agent_id)
                ):
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

        # SELECT ... FROM conversation_history WHERE id = ? AND agent_id = ? AND deleted_at IS NULL
        if "from conversation_history" in sql_lower and "where id = ?" in sql_lower:
            msg_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            msg = self.messages.get(msg_id)
            if not msg or (agent_id is not None and msg["agent_id"] != agent_id):
                return None
            if "content, metadata" in sql_lower and "id," in sql_lower:
                return (msg["id"], msg["content"], msg["metadata"])
            if "metadata" in sql_lower:
                return (msg["id"], msg["metadata"])
            return (msg["id"], msg["content"], msg["metadata"])

        # SELECT id FROM memory_pins WHERE message_id = ? AND agent_id = ? AND released_at IS NULL
        if "from memory_pins" in sql_lower and "released_at is null" in sql_lower:
            message_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            for pin in self.pins.values():
                if (
                    pin["message_id"] == message_id
                    and pin["released_at"] is None
                    and (agent_id is None or pin["agent_id"] == agent_id)
                ):
                    return (pin["id"],)
            return None

        return None

    # -- multi-row reads ---------------------------------------------------

    async def fetchall(self, sql, params=()):
        sql_lower = sql.strip().lower()

        # JOIN query for memory_pinned() -- WHERE ch.agent_id = ? AND ch.deleted_at IS NULL
        if "from memory_pins" in sql_lower and "join conversation_history" in sql_lower:
            agent_id = params[0] if params else None
            results = []
            for pin in self.pins.values():
                if pin["released_at"] is not None:
                    continue
                msg = self.messages.get(pin["message_id"])
                if msg and (agent_id is None or msg["agent_id"] == agent_id):
                    results.append((
                        pin["id"],
                        pin["message_id"],
                        pin["pin_reason"],
                        pin["pinned_at"],
                        msg["content"],
                    ))
            return results

        # SELECT message_id FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        # (used by admin_unpin_all)
        if (
            "select message_id from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            agent_id = params[0] if params else None
            return [
                (p["message_id"],)
                for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]

        # SELECT id, message_id FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        # ORDER BY pinned_at ASC LIMIT ?
        # (used by admin_unpin_oldest)
        if (
            "select id, message_id from memory_pins" in sql_lower
            and "order by pinned_at asc" in sql_lower
        ):
            agent_id = params[0] if params else None
            limit = params[1] if len(params) > 1 else None
            active = sorted(
                [
                    p for p in self.pins.values()
                    if p["released_at"] is None
                    and (agent_id is None or p["agent_id"] == agent_id)
                ],
                key=lambda p: p["pinned_at"],
            )
            if limit is not None:
                active = active[:limit]
            return [(p["id"], p["message_id"]) for p in active]

        # SELECT pinned_at FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        # (used by stats average pin age)
        if (
            "select pinned_at from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            agent_id = params[0] if params else None
            return [
                (p["pinned_at"],)
                for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
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

        # COUNT(*) on memory_pins WHERE agent_id = ? AND released_at IS [NOT] NULL
        if "count(*)" in sql_lower and "memory_pins" in sql_lower:
            agent_id = params[0] if params else None
            scoped = [
                p for p in self.pins.values()
                if agent_id is None or p["agent_id"] == agent_id
            ]
            if "released_at is null" in sql_lower:
                return sum(1 for p in scoped if p["released_at"] is None)
            if "released_at is not null" in sql_lower:
                return sum(1 for p in scoped if p["released_at"] is not None)
            return len(scoped)

        # MIN(pinned_at) FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        if "min(pinned_at)" in sql_lower:
            agent_id = params[0] if params else None
            active = [
                p["pinned_at"] for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]
            return min(active) if active else None

        # MAX(pinned_at) FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        if "max(pinned_at)" in sql_lower:
            agent_id = params[0] if params else None
            active = [
                p["pinned_at"] for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]
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
    feature._db = fake_db
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
        assert result.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL), (
            f"Pin {i} should succeed (got {result.status})"
        )
        assert result.data["pinned"] is True

    # The 101st pin should be rejected
    result = await feature.memory_pin(
        message_id=msg_ids[PIN_QUOTA_DEFAULT], reason="one too many"
    )
    assert result.status is ToolResultStatus.ERROR
    assert "quota" in result.error.lower()


@pytest.mark.asyncio
async def test_pin_quota_configurable():
    """Setting a custom quota should be enforced at that limit."""
    db = FakeDB()
    custom_quota = 5
    feature = _make_feature(db, pin_quota=custom_quota)

    msg_ids = [db.add_message(f"Message {i}") for i in range(custom_quota + 1)]

    for i in range(custom_quota):
        result = await feature.memory_pin(message_id=msg_ids[i])
        assert result.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)
        assert result.data["pinned"] is True

    # The (custom_quota + 1)th pin should fail
    result = await feature.memory_pin(message_id=msg_ids[custom_quota])
    assert result.status is ToolResultStatus.ERROR
    assert str(custom_quota) in result.error


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
    assert result.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)
    assert result.data["pinned"] is True

    # A genuinely new pin should still be rejected
    result = await feature.memory_pin(message_id=msg_ids[2])
    assert result.status is ToolResultStatus.ERROR


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

    # Honesty: high pin ratio surfaces as PARTIAL with the over-pinning
    # caveat in result.error (was result["warning"] pre-#1042 layer 4).
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["pinned"] is True
    assert "ratio" in result.error.lower()


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

    # 3/10 = 30% ratio, below threshold → OK (no over-pinning caveat).
    assert result.status is ToolResultStatus.OK
    assert result.data["pinned"] is True
    assert not result.error


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

    assert result.status is ToolResultStatus.OK
    assert result.data["unpinned"] == 5

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
    assert result.status is ToolResultStatus.OK
    assert result.data["unpinned"] == 0


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

    assert result.status is ToolResultStatus.OK
    assert result.data["unpinned"] == 2
    assert result.data["requested"] == 2

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

    # Honesty: requested 10 but only 3 active → PARTIAL with shortfall caveat.
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["unpinned"] == 3
    assert result.data["requested"] == 10
    assert "10" in result.error or "3" in result.error

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

    # 30% ratio is below alert threshold → OK (no over-pinning caveat).
    assert stats.status is ToolResultStatus.OK
    assert stats.data["total_messages"] == 10
    assert stats.data["pinned"] == 3
    assert stats.data["quota"] == 50
    assert stats.data["quota_remaining"] == 47
    assert stats.data["pin_ratio"] == 0.3
    # Age fields should be present (non-None since we have pins)
    assert stats.data["oldest_pin_age_seconds"] is not None
    assert stats.data["average_pin_age_seconds"] is not None
    assert not stats.error


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

    # 75% ratio exceeds threshold → PARTIAL with alert in result.error.
    assert stats.status is ToolResultStatus.PARTIAL
    assert stats.data["pin_ratio"] == 0.75
    assert "threshold" in stats.error.lower()


@pytest.mark.asyncio
async def test_pin_stats_no_pins():
    """Pin stats with no pins should have zeroes and no alert."""
    db = FakeDB()
    feature = _make_feature(db, pin_quota=100)

    # Create some messages but pin none
    for i in range(5):
        db.add_message(f"Message {i}")

    stats = await feature.memory_pin_stats()

    assert stats.status is ToolResultStatus.OK
    assert stats.data["pinned"] == 0
    assert stats.data["quota"] == 100
    assert stats.data["quota_remaining"] == 100
    assert stats.data["pin_ratio"] == 0.0
    assert stats.data["oldest_pin_age_seconds"] is None
    assert stats.data["average_pin_age_seconds"] is None
    assert not stats.error


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
    # 100% ratio → PARTIAL with over-pinning caveat.
    assert stats.status is ToolResultStatus.PARTIAL
    assert stats.data["pinned"] == 10
    assert stats.data["quota"] == 5
    assert stats.data["quota_remaining"] == 0  # floor at zero, not negative


# --------------------------------------------------------------------------
# Multi-tenant isolation (shared DB, two agents) -- regression for #2085
#
# The pin table is shared across agents in a multi-agent deployment. Every
# memory_pins read/write MUST be scoped to self.agent_id, or one agent's
# quota, stats, and admin unpin bleed into another agent's pins.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quota_is_per_agent_not_global():
    """Agent A filling its quota must not block agent B on a shared DB."""
    db = FakeDB()
    feat_a = _make_feature(db, agent_id="agent-A", pin_quota=2)
    feat_b = _make_feature(db, agent_id="agent-B", pin_quota=2)

    # Extra unpinned messages keep the pin ratio below the warning threshold
    # so the result stays OK (not PARTIAL) and the assertion targets quota.
    for i in range(10):
        db.add_message(f"A-extra-{i}", agent_id="agent-A")
    a_msgs = [db.add_message(f"A{i}", agent_id="agent-A") for i in range(2)]
    for mid in a_msgs:
        res = await feat_a.memory_pin(message_id=mid)
        assert res.status is ToolResultStatus.OK

    # Agent A is now at quota; agent B must still be able to pin.
    for i in range(10):
        db.add_message(f"B-extra-{i}", agent_id="agent-B")
    b_msg = db.add_message("B0", agent_id="agent-B")
    res_b = await feat_b.memory_pin(message_id=b_msg)
    assert res_b.status is ToolResultStatus.OK, (
        "agent B blocked by agent A's pins -> quota not agent-scoped"
    )
    assert res_b.data["pinned"] is True


@pytest.mark.asyncio
async def test_pin_stats_counts_only_own_pins():
    """memory_pin_stats must not count another agent's pins."""
    db = FakeDB()
    feat_a = _make_feature(db, agent_id="agent-A", pin_quota=100)
    feat_b = _make_feature(db, agent_id="agent-B", pin_quota=100)

    for i in range(3):
        await feat_a.memory_pin(message_id=db.add_message(f"A{i}", agent_id="agent-A"))
    await feat_b.memory_pin(message_id=db.add_message("B0", agent_id="agent-B"))

    stats_b = await feat_b.memory_pin_stats()
    assert stats_b.data["pinned"] == 1, "stats leaked agent A's pins into agent B"


@pytest.mark.asyncio
async def test_admin_unpin_all_only_releases_own_pins():
    """memory_admin_unpin_all must leave other agents' pins active."""
    db = FakeDB()
    feat_a = _make_feature(db, agent_id="agent-A", pin_quota=100)
    feat_b = _make_feature(db, agent_id="agent-B", pin_quota=100)

    for i in range(2):
        await feat_a.memory_pin(message_id=db.add_message(f"A{i}", agent_id="agent-A"))
    b_msg = db.add_message("B0", agent_id="agent-B")
    await feat_b.memory_pin(message_id=b_msg)

    res = await feat_a.memory_admin_unpin_all()
    assert res.status is ToolResultStatus.OK

    # Agent B's pin must survive agent A's bulk unpin.
    stats_b = await feat_b.memory_pin_stats()
    assert stats_b.data["pinned"] == 1, "agent A's unpin-all released agent B's pins"


@pytest.mark.asyncio
async def test_admin_unpin_oldest_only_releases_own_pins():
    """memory_admin_unpin_oldest must ignore other agents' older pins."""
    db = FakeDB()
    feat_a = _make_feature(db, agent_id="agent-A", pin_quota=100)
    feat_b = _make_feature(db, agent_id="agent-B", pin_quota=100)

    # Agent B pins first (oldest overall), then agent A pins.
    b_msg = db.add_message("B0", agent_id="agent-B")
    await feat_b.memory_pin(message_id=b_msg)
    a_msg = db.add_message("A0", agent_id="agent-A")
    await feat_a.memory_pin(message_id=a_msg)

    # Agent A asks to unpin its 1 oldest -- must hit A's pin, not B's older one.
    res = await feat_a.memory_admin_unpin_oldest(count=1)
    assert res.status is ToolResultStatus.OK

    stats_b = await feat_b.memory_pin_stats()
    assert stats_b.data["pinned"] == 1, "unpin-oldest crossed into agent B's pins"
    stats_a = await feat_a.memory_pin_stats()
    assert stats_a.data["pinned"] == 0
