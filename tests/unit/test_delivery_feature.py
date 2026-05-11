"""
Unit tests for the DeliveryFeature, DeliveryQueue, and data models.

Tests:
- Feature initialization and tool registration
- Queue enqueue, process, retry, dead-letter, and purge operations
- Deduplication within the 60-second window
- Exponential backoff computation
- Background worker lifecycle (start/stop)
- Dead letter queue management
- Data model serialization
- Error handling for missing DB, unknown entries, etc.
"""

import json
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.delivery.feature import DeliveryFeature
from kestrel_sovereign.features.delivery.models import (
    DeliveryResult,
    DeliveryStatus,
    QueueEntry,
)
from kestrel_sovereign.features.delivery.queue import (
    DeliveryQueue,
    _compute_backoff,
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_mock_db():
    """Create a mock AsyncDatabase with standard methods."""
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=0)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_mock_agent(db=None):
    """Create a mock agent with storage.db."""
    agent = MagicMock()
    agent.agent_id = "did:test:delivery-agent"
    agent.features = {}

    mock_db = db or _make_mock_db()
    agent.storage = MagicMock()
    agent.storage.db = mock_db

    return agent


def _make_queue_row(
    entry_id="entry-1",
    agent_id="did:test:delivery-agent",
    channel_type="webhook",
    recipient="https://example.com/hook",
    content_json='{"text": "hello"}',
    content_hash="abc123",
    status="pending",
    attempts=0,
    max_retries=5,
    next_retry_at=None,
    last_error=None,
    created_at=None,
    delivered_at=None,
):
    """Create a mock queue row tuple matching the SELECT column order."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if next_retry_at is None:
        next_retry_at = created_at
    return (
        entry_id, agent_id, channel_type, recipient, content_json,
        content_hash, status, attempts, max_retries, next_retry_at,
        last_error, created_at, delivered_at,
    )


def _make_dead_letter_row(
    dl_id="dl-1",
    original_id="entry-1",
    agent_id="did:test:delivery-agent",
    channel_type="webhook",
    recipient="https://example.com/hook",
    content_json='{"text": "hello"}',
    error="Connection refused",
    attempts=5,
    created_at=None,
):
    """Create a mock dead letter row tuple."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    return (dl_id, original_id, agent_id, channel_type, recipient, content_json, error, attempts, created_at)


# =========================================================================
# Fixtures
# =========================================================================


@pytest_asyncio.fixture
async def feature():
    """Create and initialize a DeliveryFeature with mocked agent/db."""
    agent = _make_mock_agent()
    f = DeliveryFeature(agent)
    # Patch the queue so it does not actually start a background task
    with patch.object(DeliveryQueue, "start", new_callable=AsyncMock):
        await f.initialize()
    return f


@pytest_asyncio.fixture
async def feature_no_db():
    """DeliveryFeature with no database available."""
    agent = MagicMock(spec=["agent_id", "did", "features"])
    agent.agent_id = "did:test:no-db"
    agent.features = {}
    f = DeliveryFeature(agent)
    await f.initialize()
    return f


@pytest_asyncio.fixture
async def queue():
    """Create a DeliveryQueue with mocked DB (not started)."""
    db = _make_mock_db()
    q = DeliveryQueue(db, "did:test:delivery-agent")
    return q


# =========================================================================
# Backoff computation
# =========================================================================


class TestBackoffComputation:

    def test_attempt_0(self):
        # 5 * (5 ** 0) = 5 seconds
        assert _compute_backoff(0) == 5

    def test_attempt_1(self):
        # 5 * (5 ** 1) = 25 seconds
        assert _compute_backoff(1) == 25

    def test_attempt_2(self):
        # 5 * (5 ** 2) = 125 seconds = 2m 5s
        assert _compute_backoff(2) == 125

    def test_attempt_3(self):
        # 5 * (5 ** 3) = 625 seconds = 10m 25s
        assert _compute_backoff(3) == 625

    def test_attempt_4(self):
        # 5 * (5 ** 4) = 3125 seconds = 52m 5s
        assert _compute_backoff(4) == 3125

    def test_attempt_5_capped(self):
        # 5 * (5 ** 5) = 15625 > 3600, should be capped at 3600
        assert _compute_backoff(5) == MAX_DELAY_SECONDS

    def test_very_high_attempt_capped(self):
        assert _compute_backoff(100) == MAX_DELAY_SECONDS


# =========================================================================
# DeliveryStatus enum
# =========================================================================


class TestDeliveryStatusEnum:

    def test_all_values(self):
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.IN_FLIGHT.value == "in_flight"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.FAILED.value == "failed"
        assert DeliveryStatus.DEAD_LETTER.value == "dead_letter"

    def test_from_value(self):
        assert DeliveryStatus("pending") is DeliveryStatus.PENDING
        assert DeliveryStatus("dead_letter") is DeliveryStatus.DEAD_LETTER


# =========================================================================
# QueueEntry model
# =========================================================================


class TestQueueEntryModel:

    def test_content_parses_valid_json(self):
        entry = QueueEntry(
            id="e1", agent_id="a1", channel_type="webhook",
            recipient="http://example.com", content_json='{"key": "value"}',
            status=DeliveryStatus.PENDING, attempts=0, max_retries=5,
            next_retry_at=None, last_error=None,
            created_at="2026-01-01T00:00:00", delivered_at=None,
        )
        assert entry.content == {"key": "value"}

    def test_content_returns_empty_for_invalid_json(self):
        entry = QueueEntry(
            id="e1", agent_id="a1", channel_type="webhook",
            recipient="http://example.com", content_json="not json",
            status=DeliveryStatus.PENDING, attempts=0, max_retries=5,
            next_retry_at=None, last_error=None,
            created_at="2026-01-01T00:00:00", delivered_at=None,
        )
        assert entry.content == {}

    def test_content_returns_empty_for_empty_string(self):
        entry = QueueEntry(
            id="e1", agent_id="a1", channel_type="webhook",
            recipient="http://example.com", content_json="",
            status=DeliveryStatus.PENDING, attempts=0, max_retries=5,
            next_retry_at=None, last_error=None,
            created_at="2026-01-01T00:00:00", delivered_at=None,
        )
        assert entry.content == {}

    def test_to_dict(self):
        entry = QueueEntry(
            id="e1", agent_id="a1", channel_type="webhook",
            recipient="http://example.com", content_json='{"text": "hi"}',
            status=DeliveryStatus.DELIVERED, attempts=1, max_retries=5,
            next_retry_at=None, last_error=None,
            created_at="2026-01-01T00:00:00", delivered_at="2026-01-01T00:01:00",
        )
        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["status"] == "delivered"
        assert d["content"] == {"text": "hi"}
        assert d["delivered_at"] == "2026-01-01T00:01:00"

    def test_compute_content_hash_deterministic(self):
        h1 = QueueEntry.compute_content_hash("user@test.com", '{"msg": "hello"}')
        h2 = QueueEntry.compute_content_hash("user@test.com", '{"msg": "hello"}')
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_content_hash_differs_by_recipient(self):
        h1 = QueueEntry.compute_content_hash("user1@test.com", '{"msg": "hello"}')
        h2 = QueueEntry.compute_content_hash("user2@test.com", '{"msg": "hello"}')
        assert h1 != h2

    def test_compute_content_hash_differs_by_content(self):
        h1 = QueueEntry.compute_content_hash("user@test.com", '{"msg": "hello"}')
        h2 = QueueEntry.compute_content_hash("user@test.com", '{"msg": "world"}')
        assert h1 != h2


# =========================================================================
# DeliveryResult model
# =========================================================================


class TestDeliveryResult:

    def test_success_result(self):
        r = DeliveryResult(success=True)
        d = r.to_dict()
        assert d == {"success": True}

    def test_failure_result(self):
        r = DeliveryResult(success=False, error="Connection timeout")
        d = r.to_dict()
        assert d == {"success": False, "error": "Connection timeout"}


# =========================================================================
# Feature tool registration
# =========================================================================


class TestDeliveryToolRegistration:

    @pytest.mark.asyncio
    async def test_feature_has_correct_tools(self, feature):
        tools = feature.get_tools()
        tool_names = {t.name for t in tools}
        assert "delivery_status" in tool_names
        assert "delivery_queue_list" in tool_names
        assert "delivery_failed" in tool_names
        assert "delivery_retry" in tool_names
        assert "delivery_purge" in tool_names

    @pytest.mark.asyncio
    async def test_tool_count(self, feature):
        tools = feature.get_tools()
        assert len(tools) == 5

    @pytest.mark.asyncio
    async def test_tool_description(self, feature):
        desc = feature.tool_description
        assert "delivery" in desc.lower()


# =========================================================================
# Feature initialization
# =========================================================================


class TestDeliveryInit:

    @pytest.mark.asyncio
    async def test_initialize_without_storage(self):
        agent = MagicMock(spec=["agent_id", "did", "features"])
        agent.agent_id = "did:test:no-storage"
        agent.features = {}
        f = DeliveryFeature(agent)
        await f.initialize()
        assert f._db is None
        assert f._queue is None

    @pytest.mark.asyncio
    async def test_shutdown_stops_queue(self, feature):
        feature._queue = MagicMock()
        feature._queue.stop = AsyncMock()
        await feature.shutdown()
        feature._queue.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_without_queue(self, feature_no_db):
        # Should not raise
        await feature_no_db.shutdown()


# =========================================================================
# delivery_status tool
# =========================================================================


class TestDeliveryStatusTool:

    @pytest.mark.asyncio
    async def test_status_returns_counts(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._queue.get_status_counts = AsyncMock(return_value={
            "pending": 3,
            "in_flight": 1,
            "delivered": 10,
            "failed": 2,
            "dead_letter": 0,
        })
        envelope = await feature.delivery_status()
        assert envelope.status is ToolResultStatus.OK
        result = envelope.data
        assert result["counts"]["pending"] == 3
        assert result["total"] == 16
        assert result["queue_healthy"] is True

    @pytest.mark.asyncio
    async def test_status_unhealthy_with_dead_letters(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._queue.get_status_counts = AsyncMock(return_value={
            "pending": 0,
            "in_flight": 0,
            "delivered": 5,
            "failed": 0,
            "dead_letter": 2,
        })
        envelope = await feature.delivery_status()
        # Dead-letter > 0 surfaces as PARTIAL (queue is operating but
        # messages have permanently failed).
        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["queue_healthy"] is False
        assert "dead_letter" in envelope.error

    @pytest.mark.asyncio
    async def test_status_no_queue(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.delivery_status()
        assert envelope.status is ToolResultStatus.ERROR
        assert "not available" in envelope.error.lower()


# =========================================================================
# delivery_queue_list tool
# =========================================================================


class TestDeliveryQueueList:

    @pytest.mark.asyncio
    async def test_list_empty(self, feature):
        feature._queue.get_pending_entries = AsyncMock(return_value=[])
        envelope = await feature.delivery_queue_list()
        assert envelope.data["entries"] == []
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_entries(self, feature):
        entry = QueueEntry(
            id="e1", agent_id="a1", channel_type="webhook",
            recipient="http://example.com", content_json='{"text": "hi"}',
            status=DeliveryStatus.PENDING, attempts=0, max_retries=5,
            next_retry_at="2026-01-01T00:00:00", last_error=None,
            created_at="2026-01-01T00:00:00", delivered_at=None,
        )
        feature._queue.get_pending_entries = AsyncMock(return_value=[entry])
        envelope = await feature.delivery_queue_list()
        assert envelope.data["count"] == 1
        assert envelope.data["entries"][0]["id"] == "e1"

    @pytest.mark.asyncio
    async def test_list_no_queue(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.delivery_queue_list()
        assert envelope.status is ToolResultStatus.ERROR


# =========================================================================
# delivery_failed tool
# =========================================================================


class TestDeliveryFailed:

    @pytest.mark.asyncio
    async def test_failed_empty(self, feature):
        feature._queue.get_dead_letter_entries = AsyncMock(return_value=[])
        envelope = await feature.delivery_failed()
        assert envelope.data["entries"] == []
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_failed_returns_entries(self, feature):
        dl_entry = {
            "id": "dl-1",
            "original_id": "e1",
            "agent_id": "a1",
            "channel_type": "webhook",
            "recipient": "http://example.com",
            "content": {"text": "hi"},
            "error": "Connection refused",
            "attempts": 5,
            "created_at": "2026-01-01T00:00:00",
        }
        feature._queue.get_dead_letter_entries = AsyncMock(return_value=[dl_entry])
        envelope = await feature.delivery_failed()
        assert envelope.data["count"] == 1
        assert envelope.data["entries"][0]["error"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_failed_no_queue(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.delivery_failed()
        assert envelope.status is ToolResultStatus.ERROR


# =========================================================================
# delivery_retry tool
# =========================================================================


class TestDeliveryRetry:

    @pytest.mark.asyncio
    async def test_retry_success(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._queue.retry = AsyncMock(return_value={
            "success": True,
            "entry_id": "e1",
            "status": "queued_for_retry",
        })
        envelope = await feature.delivery_retry(message_id="e1")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["entry_id"] == "e1"

    @pytest.mark.asyncio
    async def test_retry_not_found(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._queue.retry = AsyncMock(return_value={
            "success": False,
            "error": "Entry e1 not found",
        })
        envelope = await feature.delivery_retry(message_id="e1")
        assert envelope.status is ToolResultStatus.ERROR
        assert "not found" in envelope.error

    @pytest.mark.asyncio
    async def test_retry_no_queue(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.delivery_retry(message_id="e1")
        assert envelope.status is ToolResultStatus.ERROR


# =========================================================================
# delivery_purge tool
# =========================================================================


class TestDeliveryPurge:

    @pytest.mark.asyncio
    async def test_purge_success(self, feature):
        from kestrel_sdk.tools.result import ToolResultStatus
        feature._queue.purge_delivered = AsyncMock(return_value=5)
        envelope = await feature.delivery_purge()
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["purged"] == 5
        assert envelope.data["older_than_hours"] == 24

    @pytest.mark.asyncio
    async def test_purge_custom_hours(self, feature):
        feature._queue.purge_delivered = AsyncMock(return_value=2)
        envelope = await feature.delivery_purge(older_than_hours=48)
        assert envelope.data["purged"] == 2
        assert envelope.data["older_than_hours"] == 48

    @pytest.mark.asyncio
    async def test_purge_no_queue(self, feature_no_db):
        from kestrel_sdk.tools.result import ToolResultStatus
        envelope = await feature_no_db.delivery_purge()
        assert envelope.status is ToolResultStatus.ERROR


# =========================================================================
# DeliveryQueue - table creation
# =========================================================================


class TestQueueTableCreation:

    @pytest.mark.asyncio
    async def test_ensure_tables_creates_tables_and_indexes(self, queue):
        await queue._ensure_tables()
        # 2 tables + 3 indexes = 5 execute calls
        assert queue._db.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_ensure_tables_includes_delivery_queue(self, queue):
        await queue._ensure_tables()
        calls = [str(c) for c in queue._db.execute.call_args_list]
        sql_texts = [queue._db.execute.call_args_list[i][0][0] for i in range(queue._db.execute.call_count)]
        assert any("delivery_queue" in sql and "CREATE TABLE" in sql for sql in sql_texts)

    @pytest.mark.asyncio
    async def test_ensure_tables_includes_dead_letter(self, queue):
        await queue._ensure_tables()
        sql_texts = [queue._db.execute.call_args_list[i][0][0] for i in range(queue._db.execute.call_count)]
        assert any("delivery_dead_letter" in sql and "CREATE TABLE" in sql for sql in sql_texts)


# =========================================================================
# DeliveryQueue - enqueue
# =========================================================================


class TestQueueEnqueue:

    @pytest.mark.asyncio
    async def test_enqueue_creates_entry(self, queue):
        # No existing duplicate
        queue._db.fetchone = AsyncMock(return_value=None)

        entry_id = await queue.enqueue("webhook", "http://example.com", {"msg": "hi"})
        assert entry_id is not None
        assert len(entry_id) == 36  # UUID format

        # Verify INSERT was called
        insert_call = queue._db.execute.call_args
        assert "INSERT INTO delivery_queue" in insert_call[0][0]

    @pytest.mark.asyncio
    async def test_enqueue_deduplicates(self, queue):
        # Simulate existing entry within dedup window
        queue._db.fetchone = AsyncMock(return_value=("existing-id",))

        entry_id = await queue.enqueue("webhook", "http://example.com", {"msg": "hi"})
        assert entry_id == "existing-id"

        # No INSERT should have been called (only the dedup SELECT)
        queue._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_enqueue_custom_max_retries(self, queue):
        queue._db.fetchone = AsyncMock(return_value=None)

        await queue.enqueue("webhook", "http://example.com", {"msg": "hi"}, max_retries=10)

        insert_call = queue._db.execute.call_args
        params = insert_call[0][1]
        # max_retries is the 8th parameter (index 7 in the tuple)
        # Parameters: id, agent_id, channel_type, recipient, content_json,
        #             content_hash, status, max_retries, next_retry_at, created_at
        assert 10 in params


# =========================================================================
# DeliveryQueue - process_pending
# =========================================================================


class TestQueueProcessPending:

    @pytest.mark.asyncio
    async def test_process_no_pending(self, queue):
        queue._db.fetchall = AsyncMock(return_value=[])
        processed = await queue.process_pending()
        assert processed == 0

    @pytest.mark.asyncio
    async def test_process_without_delivery_provider_schedules_retry(self, queue):
        row = _make_queue_row(status="pending")
        queue._db.fetchall = AsyncMock(return_value=[row])
        queue._db.fetchone = AsyncMock(return_value=None)

        processed = await queue.process_pending()
        assert processed == 1

        # Should have set status to in_flight, then failed/retryable.
        execute_calls = queue._db.execute.call_args_list
        statuses = []
        for call in execute_calls:
            sql = call[0][0]
            if "UPDATE delivery_queue" in sql and "status" in sql:
                params = call[0][1]
                statuses.append(params[0])  # status is first param
        assert "in_flight" in statuses
        assert "failed" in statuses
        assert "delivered" not in statuses

    @pytest.mark.asyncio
    async def test_process_noop_delivery_requires_explicit_opt_in(self, queue):
        row = _make_queue_row(status="pending")
        queue._db.fetchall = AsyncMock(return_value=[row])
        queue._db.fetchone = AsyncMock(return_value=None)
        queue._allow_noop_delivery = True

        processed = await queue.process_pending()
        assert processed == 1

        execute_calls = queue._db.execute.call_args_list
        statuses = [
            call[0][1][0]
            for call in execute_calls
            if "UPDATE delivery_queue" in call[0][0] and "status" in call[0][0]
        ]
        assert "delivered" in statuses

    @pytest.mark.asyncio
    async def test_process_with_delivery_callback_success(self, queue):
        row = _make_queue_row(status="pending")
        queue._db.fetchall = AsyncMock(return_value=[row])
        queue._db.fetchone = AsyncMock(return_value=None)

        deliver_fn = AsyncMock(return_value=DeliveryResult(success=True))
        queue._deliver = deliver_fn

        processed = await queue.process_pending()
        assert processed == 1
        deliver_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_delivery_callback_failure_schedules_retry(self, queue):
        row = _make_queue_row(status="pending", attempts=0, max_retries=5)
        queue._db.fetchall = AsyncMock(return_value=[row])
        queue._db.fetchone = AsyncMock(return_value=None)

        deliver_fn = AsyncMock(return_value=DeliveryResult(success=False, error="timeout"))
        queue._deliver = deliver_fn

        processed = await queue.process_pending()
        assert processed == 1

        # Should have set status to failed with next_retry_at
        execute_calls = queue._db.execute.call_args_list
        found_failed = False
        for call in execute_calls:
            sql = call[0][0]
            if "UPDATE delivery_queue" in sql:
                params = call[0][1]
                if params[0] == "failed" and "next_retry_at" in sql:
                    found_failed = True
                    assert params[1] == 1  # attempts incremented
        assert found_failed

    @pytest.mark.asyncio
    async def test_process_max_retries_dead_letters(self, queue):
        # Already at max_retries - 1 attempts, so next failure triggers DLQ
        row = _make_queue_row(status="failed", attempts=4, max_retries=5)
        queue._db.fetchall = AsyncMock(return_value=[row])

        # For dead-lettering: fetchone returns the row data for move_to_dead_letter
        queue._db.fetchone = AsyncMock(return_value=(
            "entry-1", "did:test:delivery-agent", "webhook",
            "https://example.com/hook", '{"text": "hello"}', 5,
            datetime.now(timezone.utc).isoformat(),
        ))

        deliver_fn = AsyncMock(return_value=DeliveryResult(success=False, error="permanent failure"))
        queue._deliver = deliver_fn

        processed = await queue.process_pending()
        assert processed == 1

        # Should have INSERTed into delivery_dead_letter
        execute_calls = queue._db.execute.call_args_list
        dl_inserts = [c for c in execute_calls if "delivery_dead_letter" in str(c[0][0]) and "INSERT" in str(c[0][0])]
        assert len(dl_inserts) >= 1

    @pytest.mark.asyncio
    async def test_process_delivery_exception_treated_as_failure(self, queue):
        row = _make_queue_row(status="pending", attempts=0, max_retries=5)
        queue._db.fetchall = AsyncMock(return_value=[row])
        queue._db.fetchone = AsyncMock(return_value=None)

        deliver_fn = AsyncMock(side_effect=ConnectionError("network down"))
        queue._deliver = deliver_fn

        processed = await queue.process_pending()
        assert processed == 1

        # Should have recorded the exception as an error
        execute_calls = queue._db.execute.call_args_list
        found_error = False
        for call in execute_calls:
            params = call[0][1] if len(call[0]) > 1 else ()
            if any("network down" in str(p) for p in params):
                found_error = True
        assert found_error


# =========================================================================
# DeliveryQueue - retry
# =========================================================================


class TestQueueRetry:

    @pytest.mark.asyncio
    async def test_retry_failed_entry(self, queue):
        row = _make_queue_row(entry_id="e1", status="failed")
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is True
        assert result["status"] == "queued_for_retry"

    @pytest.mark.asyncio
    async def test_retry_already_delivered(self, queue):
        row = _make_queue_row(entry_id="e1", status="delivered")
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is False
        assert "already delivered" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_retry_in_flight(self, queue):
        row = _make_queue_row(entry_id="e1", status="in_flight")
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is False
        assert "in flight" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_retry_from_dead_letter(self, queue):
        # First call (main queue lookup) returns None
        # Second call (dead letter lookup) returns a dead letter row
        dl_row = _make_dead_letter_row(dl_id="dl-1", original_id="e1")
        queue._db.fetchone = AsyncMock(side_effect=[None, dl_row])

        result = await queue.retry("e1")
        assert result["success"] is True
        assert result["status"] == "re-enqueued_from_dead_letter"

        # Should have INSERTed into delivery_queue and DELETEd from dead_letter
        execute_calls = queue._db.execute.call_args_list
        assert any("INSERT INTO delivery_queue" in str(c) for c in execute_calls)
        assert any("DELETE FROM delivery_dead_letter" in str(c) for c in execute_calls)

    @pytest.mark.asyncio
    async def test_retry_not_found(self, queue):
        queue._db.fetchone = AsyncMock(return_value=None)

        result = await queue.retry("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =========================================================================
# DeliveryQueue - move_to_dead_letter
# =========================================================================


class TestMoveToDeadLetter:

    @pytest.mark.asyncio
    async def test_move_to_dead_letter(self, queue):
        queue._db.fetchone = AsyncMock(return_value=(
            "e1", "did:test:delivery-agent", "webhook",
            "http://example.com", '{"msg": "hi"}', 5,
            datetime.now(timezone.utc).isoformat(),
        ))

        await queue.move_to_dead_letter("e1", "Max retries exceeded")

        execute_calls = queue._db.execute.call_args_list
        # Should INSERT into dead_letter and DELETE from main queue
        assert any("delivery_dead_letter" in str(c) and "INSERT" in str(c) for c in execute_calls)
        assert any("DELETE FROM delivery_queue" in str(c) for c in execute_calls)

    @pytest.mark.asyncio
    async def test_move_to_dead_letter_unknown_entry(self, queue):
        queue._db.fetchone = AsyncMock(return_value=None)
        # Should not raise, just log a warning
        await queue.move_to_dead_letter("nonexistent", "test")
        queue._db.execute.assert_not_called()


# =========================================================================
# DeliveryQueue - status counts
# =========================================================================


class TestStatusCounts:

    @pytest.mark.asyncio
    async def test_status_counts_all_statuses(self, queue):
        queue._db.fetchall = AsyncMock(return_value=[
            ("pending", 3),
            ("delivered", 10),
            ("failed", 2),
            ("in_flight", 1),
        ])
        queue._db.fetchone = AsyncMock(return_value=(5,))

        counts = await queue.get_status_counts()
        assert counts["pending"] == 3
        assert counts["delivered"] == 10
        assert counts["failed"] == 2
        assert counts["in_flight"] == 1
        assert counts["dead_letter"] == 5

    @pytest.mark.asyncio
    async def test_status_counts_empty(self, queue):
        queue._db.fetchall = AsyncMock(return_value=[])
        queue._db.fetchone = AsyncMock(return_value=(0,))

        counts = await queue.get_status_counts()
        assert counts["pending"] == 0
        assert counts["dead_letter"] == 0


# =========================================================================
# DeliveryQueue - purge
# =========================================================================


class TestQueuePurge:

    @pytest.mark.asyncio
    async def test_purge_delivered(self, queue):
        queue._db.fetchone = AsyncMock(return_value=(7,))

        purged = await queue.purge_delivered(older_than_hours=24)
        assert purged == 7

        # Should have called DELETE
        execute_calls = queue._db.execute.call_args_list
        assert any("DELETE FROM delivery_queue" in str(c) for c in execute_calls)

    @pytest.mark.asyncio
    async def test_purge_nothing_to_purge(self, queue):
        queue._db.fetchone = AsyncMock(return_value=(0,))

        purged = await queue.purge_delivered()
        assert purged == 0

        # Should NOT have called DELETE (count was 0)
        execute_calls = queue._db.execute.call_args_list
        assert not any("DELETE" in str(c) for c in execute_calls)


# =========================================================================
# DeliveryQueue - get_pending_entries
# =========================================================================


class TestGetPendingEntries:

    @pytest.mark.asyncio
    async def test_returns_entries(self, queue):
        rows = [
            _make_queue_row(entry_id="e1", status="pending"),
            _make_queue_row(entry_id="e2", status="failed", attempts=2),
        ]
        queue._db.fetchall = AsyncMock(return_value=rows)

        entries = await queue.get_pending_entries()
        assert len(entries) == 2
        assert entries[0].id == "e1"
        assert entries[1].attempts == 2

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, queue):
        queue._db.fetchall = AsyncMock(return_value=[])
        entries = await queue.get_pending_entries()
        assert entries == []


# =========================================================================
# DeliveryQueue - get_dead_letter_entries
# =========================================================================


class TestGetDeadLetterEntries:

    @pytest.mark.asyncio
    async def test_returns_dead_letters(self, queue):
        rows = [_make_dead_letter_row()]
        queue._db.fetchall = AsyncMock(return_value=rows)

        entries = await queue.get_dead_letter_entries()
        assert len(entries) == 1
        assert entries[0]["error"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_handles_invalid_json_in_dead_letter(self, queue):
        rows = [_make_dead_letter_row(content_json="not valid json")]
        queue._db.fetchall = AsyncMock(return_value=rows)

        entries = await queue.get_dead_letter_entries()
        assert len(entries) == 1
        assert entries[0]["content"] == {}


# =========================================================================
# DeliveryQueue - lifecycle
# =========================================================================


class TestQueueLifecycle:

    @pytest.mark.asyncio
    async def test_start_creates_tables_and_task(self, queue):
        with patch.object(queue, "_ensure_tables", new_callable=AsyncMock) as mock_tables:
            with patch("asyncio.create_task") as mock_create:
                mock_create.return_value = MagicMock()
                await queue.start()
                mock_tables.assert_called_once()
                mock_create.assert_called_once()
                assert queue._running is True

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, queue):
        import asyncio

        # Create a real asyncio task that we can cancel
        async def noop():
            await asyncio.sleep(3600)

        loop = asyncio.get_event_loop()
        real_task = asyncio.create_task(noop())
        queue._task = real_task
        queue._running = True

        await queue.stop()
        assert queue._running is False
        assert real_task.cancelled()


# =========================================================================
# Feature - enqueue_message programmatic API
# =========================================================================


class TestEnqueueMessage:

    @pytest.mark.asyncio
    async def test_enqueue_via_feature(self, feature):
        feature._queue.enqueue = AsyncMock(return_value="new-entry-id")

        entry_id = await feature.enqueue_message(
            channel_type="webhook",
            recipient="http://example.com/hook",
            content={"text": "hello"},
        )
        assert entry_id == "new-entry-id"

    @pytest.mark.asyncio
    async def test_enqueue_without_queue(self, feature_no_db):
        entry_id = await feature_no_db.enqueue_message(
            channel_type="webhook",
            recipient="http://example.com",
            content={"text": "hello"},
        )
        assert entry_id is None


# =========================================================================
# Row-to-entry conversion
# =========================================================================


class TestRowToEntry:

    def test_row_to_entry_converts_correctly(self):
        row = _make_queue_row(
            entry_id="e1",
            status="pending",
            attempts=2,
            max_retries=5,
            last_error="timeout",
        )
        entry = DeliveryQueue._row_to_entry(row)
        assert entry.id == "e1"
        assert entry.status == DeliveryStatus.PENDING
        assert entry.attempts == 2
        assert entry.max_retries == 5
        assert entry.last_error == "timeout"

    def test_row_to_entry_handles_none_content(self):
        row = _make_queue_row(content_json=None)
        entry = DeliveryQueue._row_to_entry(row)
        assert entry.content_json == "{}"
        assert entry.content == {}


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
