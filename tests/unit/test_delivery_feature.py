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

import asyncio
import hashlib
import json
import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.features.delivery.feature import DeliveryFeature
from kestrel_sovereign.features.delivery.models import (
    DeliveryResult,
    DeliveryStatus,
    QueueEntry,
)
from kestrel_sovereign.features.delivery.queue import (
    DeliveryIdempotencyConflict,
    DeliveryIdempotencyStateError,
    DeliveryIdempotencyTerminal,
    DeliveryQueue,
    _compute_backoff,
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
)
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError


# =========================================================================
# Helpers
# =========================================================================


def _make_mock_db():
    """Create a mock AsyncDatabase with standard methods."""
    db = MagicMock()
    db.execute = AsyncMock(return_value=1)
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    db.fetchval = AsyncMock(return_value=0)
    db.backend_type = "sqlite"
    db.nested_transaction_strategy = "joined"
    db.column_exists = AsyncMock(return_value=True)
    db.column_accepts_null = AsyncMock(return_value=True)
    db.column_has_default = AsyncMock(return_value=False)

    @asynccontextmanager
    async def migration_lock(_name):
        yield

    db.migration_lock = migration_lock

    @asynccontextmanager
    async def transaction(*, immediate=False):
        del immediate
        yield

    db.transaction = transaction
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
    max_retries=5,
    retry_entry_id=None,
    legacy_content_hash=None,
):
    """Create a mock dead letter row tuple."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    return (
        dl_id,
        original_id,
        agent_id,
        channel_type,
        recipient,
        content_json,
        error,
        attempts,
        created_at,
        max_retries,
        retry_entry_id,
        legacy_content_hash,
    )


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
        # 3 tables + 11 indexes + the one-time v2 trigger cleanup + replacement
        # of the scoped SQLite atomic-compensation trigger. The v2 index is not
        # rebuilt on an already-v3 schema.
        assert queue._db.execute.call_count == 17

    @pytest.mark.asyncio
    async def test_schema_bootstrap_uses_shared_migration_lock(self, queue):
        entered = False

        @asynccontextmanager
        async def migration_lock(name):
            nonlocal entered
            assert name == "delivery_queue_schema_v3"
            entered = True
            yield

        queue._db.migration_lock = migration_lock

        await queue._ensure_tables()

        assert entered

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

    @pytest.mark.asyncio
    async def test_postgres_upgrade_reuses_shared_nullability_probe(self, queue):
        queue._db.backend_type = "postgres"
        queue._db.nested_transaction_strategy = "savepoint"
        queue._db.column_accepts_null.return_value = False

        await queue._ensure_tables()

        queue._db.column_accepts_null.assert_awaited_once_with(
            "delivery_dead_letter", "max_retries"
        )
        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "ALTER COLUMN max_retries DROP NOT NULL" in sql
        assert "pg_attribute" not in sql

    @pytest.mark.asyncio
    async def test_postgres_upgrade_skips_redundant_nullability_alter(self, queue):
        queue._db.backend_type = "postgres"
        queue._db.nested_transaction_strategy = "savepoint"
        queue._db.column_accepts_null.return_value = True

        await queue._ensure_tables()

        queue._db.column_accepts_null.assert_awaited_once_with(
            "delivery_dead_letter", "max_retries"
        )
        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "ALTER COLUMN max_retries DROP NOT NULL" not in sql
        assert "ALTER COLUMN max_retries DROP DEFAULT" not in sql

    @pytest.mark.asyncio
    async def test_postgres_upgrade_drops_prerelease_policy_default(self, queue):
        queue._db.backend_type = "postgres"
        queue._db.nested_transaction_strategy = "savepoint"
        queue._db.column_has_default.return_value = True

        await queue._ensure_tables()

        queue._db.column_has_default.assert_awaited_once_with(
            "delivery_dead_letter", "max_retries"
        )
        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "ALTER COLUMN max_retries DROP DEFAULT" in sql


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


class TestQueueIdempotency:
    def test_public_package_exports_idempotency_errors(self):
        from kestrel_sovereign.features.delivery import (
            DeliveryIdempotencyConflict as PublicConflict,
            DeliveryIdempotencyStateError as PublicStateError,
            DeliveryIdempotencyTerminal as PublicTerminal,
        )

        assert PublicConflict is DeliveryIdempotencyConflict
        assert PublicStateError is DeliveryIdempotencyStateError
        assert PublicTerminal is DeliveryIdempotencyTerminal

    @pytest_asyncio.fixture
    async def real_queue(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = await AsyncDatabase.sqlite(str(tmp_path / "idempotency.db"))
        deliveries = []

        async def deliver(channel_type, recipient, content):
            deliveries.append((channel_type, recipient, content))
            return DeliveryResult(success=True)

        queue = DeliveryQueue(
            database,
            "did:test:idempotent-owner",
            deliver=deliver,
        )
        await queue._ensure_tables()
        yield queue, deliveries
        await queue.stop()
        await database.close()

    @pytest.mark.asyncio
    async def test_unkeyed_enqueue_preserves_mixed_json_key_compatibility(
        self, real_queue
    ):
        queue, _ = real_queue

        entry_id = await queue.enqueue(
            "email",
            "mixed-keys@example.com",
            {"nested": {1: "one", "2": "two"}},
        )

        assert entry_id
        assert await queue.enqueue(
            "email",
            "mixed-keys@example.com",
            {"nested": {"2": "two", 1: "one"}},
        ) == entry_id

    @pytest.mark.asyncio
    async def test_concurrent_replay_creates_one_row_and_one_delivery(
        self, real_queue
    ):
        queue, deliveries = real_queue
        entry_ids = await asyncio.gather(
            *(
                queue.enqueue(
                    "email",
                    "person@example.com",
                    {"subject": "Escalation", "body": "Please check in"},
                    idempotency_key="workflow/run/stage/attempt",
                )
                for _ in range(20)
            )
        )

        assert len(set(entry_ids)) == 1
        row = await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        )
        assert row == (1,)

        assert await queue.process_pending() == 1
        assert len(deliveries) == 1

    @pytest.mark.asyncio
    async def test_replay_returns_canonical_id_after_delivery(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "person@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "stable-stage-key",
        }
        original_id = await queue.enqueue(**request)
        await queue._db.execute(
            "UPDATE delivery_queue SET status = ? WHERE id = ?",
            (DeliveryStatus.DELIVERED.value, original_id),
        )

        assert await queue.enqueue(**request) == original_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "changed",
        [
            {
                "channel_type": "webhook",
                "recipient": "person@example.com",
                "content": {"body": "original"},
                "max_retries": None,
            },
            {
                "channel_type": "email",
                "recipient": "other@example.com",
                "content": {"body": "original"},
                "max_retries": None,
            },
            {
                "channel_type": "email",
                "recipient": "person@example.com",
                "content": {"body": "changed"},
                "max_retries": None,
            },
            {
                "channel_type": "email",
                "recipient": "person@example.com",
                "content": {"body": "original"},
                "max_retries": 9,
            },
        ],
    )
    async def test_changed_request_fails_closed(self, real_queue, changed):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "original"},
            idempotency_key="one-logical-send",
        )

        with pytest.raises(DeliveryIdempotencyConflict):
            await queue.enqueue(
                changed["channel_type"],
                changed["recipient"],
                changed["content"],
                max_retries=changed["max_retries"],
                idempotency_key="one-logical-send",
            )

        rows = await queue._db.fetchall(
            "SELECT id, content_json FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        )
        assert rows == [(original_id, '{"body":"original"}')]

    @pytest.mark.asyncio
    async def test_mapping_order_does_not_change_logical_request(self, real_queue):
        queue, _ = real_queue
        first = await queue.enqueue(
            "email",
            "person@example.com",
            {"subject": "hello", "body": "world"},
            idempotency_key="canonical-content",
        )

        replay = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "world", "subject": "hello"},
            idempotency_key="canonical-content",
        )

        assert replay == first

    @pytest.mark.asyncio
    async def test_omitted_retry_default_remains_stable_after_restart(
        self, real_queue
    ):
        queue, _ = real_queue
        first = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="stable-omitted-default",
        )

        restarted = DeliveryQueue(
            queue._db,
            queue._agent_id,
            max_retries=99,
        )
        replay = await restarted.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="stable-omitted-default",
        )

        assert replay == first

    @pytest.mark.asyncio
    async def test_stale_claim_repair_preserves_original_effective_retries(
        self, real_queue
    ):
        queue, _ = real_queue
        original = await queue.enqueue(
            "email",
            "stale-policy@example.com",
            {"body": "hello"},
            idempotency_key="stale-policy",
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original, queue._agent_id),
        )

        restarted = DeliveryQueue(queue._db, queue._agent_id, max_retries=99)
        repaired = await restarted.enqueue(
            "email",
            "stale-policy@example.com",
            {"body": "hello"},
            idempotency_key="stale-policy",
        )

        assert repaired != original
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?", (repaired,)
        ) == (queue._max_retries,)

    @pytest.mark.asyncio
    async def test_fresh_claim_skips_stale_compatibility_probe(self, real_queue):
        queue, _ = real_queue

        with patch.object(
            queue,
            "_has_unlinked_compatible_queue_row",
            new_callable=AsyncMock,
        ) as compatibility_probe:
            entry_id = await queue.enqueue(
                "email",
                "fresh-claim@example.com",
                {"body": "hello"},
                idempotency_key="fresh-claim",
            )

        assert entry_id
        compatibility_probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pre_upgrade_stale_claim_without_policy_fails_closed(
        self, real_queue
    ):
        queue, _ = real_queue
        original = await queue.enqueue(
            "email",
            "unknown-policy@example.com",
            {"body": "hello"},
            idempotency_key="unknown-policy",
        )
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET effective_max_retries = NULL
            WHERE agent_id = ?
            """,
            (queue._agent_id,),
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original, queue._agent_id),
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="retry policy"):
            await queue.enqueue(
                "email",
                "unknown-policy@example.com",
                {"body": "hello"},
                idempotency_key="unknown-policy",
            )

    @pytest.mark.asyncio
    async def test_policyless_alias_stays_fail_closed_after_peer_repair(
        self, real_queue
    ):
        queue, _ = real_queue
        request = ("email", "policyless-alias@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="policyless-alias-a"
        )
        assert await queue.enqueue(
            *request, idempotency_key="policyless-alias-b", max_retries=5
        ) == original_id
        missing_policy_digest = hashlib.sha256(
            b"policyless-alias-a"
        ).hexdigest()
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET effective_max_retries = NULL
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (queue._agent_id, missing_policy_digest),
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        replacement_id = await queue.enqueue(
            *request, idempotency_key="policyless-alias-b", max_retries=5
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="retry policy"):
            await queue.enqueue(
                *request, idempotency_key="policyless-alias-a"
            )
        assert await queue._db.fetchone(
            """
            SELECT effective_max_retries, previous_entry_id
            FROM delivery_idempotency
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (queue._agent_id, missing_policy_digest),
        ) == (None, original_id)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (replacement_id, queue._agent_id),
        ) == (1,)

    @pytest.mark.asyncio
    async def test_linked_queue_row_is_not_unlinked_compatibility_hazard(
        self, real_queue
    ):
        queue, _ = real_queue
        entry_id = await queue.enqueue(
            "email",
            "linked-compatible@example.com",
            {"body": "hello"},
            idempotency_key="linked-compatible",
        )
        row = await queue._db.fetchone(
            """
            SELECT recipient, canonical_content_hash, content_hash,
                   channel_type, created_at
            FROM delivery_queue WHERE id = ? AND agent_id = ?
            """,
            (entry_id, queue._agent_id),
        )

        assert not await queue._has_unlinked_compatible_queue_row(
            recipient=row[0],
            canonical_content_hash=row[1],
            legacy_content_hash=row[2],
            channel_type=row[3],
            claim_created_at=row[4],
        )

    @pytest.mark.asyncio
    async def test_pre_upgrade_live_claim_backfills_replay_metadata(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "live-upgrade@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "live-upgrade",
        }
        entry_id = await queue.enqueue(**request)
        content_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (entry_id,)
        )
        await queue._db.execute(
            """
            UPDATE delivery_idempotency
            SET effective_max_retries = NULL, legacy_content_hash = NULL
            WHERE agent_id = ?
            """,
            (queue._agent_id,),
        )

        assert await queue.enqueue(**request) == entry_id
        assert await queue._db.fetchone(
            """
            SELECT effective_max_retries, legacy_content_hash
            FROM delivery_idempotency WHERE agent_id = ?
            """,
            (queue._agent_id,),
        ) == (queue._max_retries, content_hash[0])

    @pytest.mark.asyncio
    async def test_explicit_default_is_distinct_from_omitted_default(
        self, real_queue
    ):
        queue, _ = real_queue
        await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="omission-is-request-identity",
        )

        with pytest.raises(DeliveryIdempotencyConflict):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"body": "hello"},
                max_retries=queue._max_retries,
                idempotency_key="omission-is-request-identity",
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["", "x" * 4097])
    async def test_invalid_idempotency_key_is_rejected(self, real_queue, key):
        queue, _ = real_queue

        with pytest.raises(ValueError):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"body": "hello"},
                idempotency_key=key,
            )

    @pytest.mark.asyncio
    async def test_same_key_is_independent_between_owners(self, real_queue):
        queue, _ = real_queue
        other = DeliveryQueue(queue._db, "did:test:other-owner")
        request = (
            "email",
            "person@example.com",
            {"body": "hello"},
        )

        mine = await queue.enqueue(*request, idempotency_key="shared-key")
        theirs = await other.enqueue(*request, idempotency_key="shared-key")

        assert mine != theirs

    @pytest.mark.asyncio
    async def test_only_key_digest_is_persisted(self, real_queue):
        queue, _ = real_queue
        raw_key = "workflow-identity-must-not-be-stored"
        await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key=raw_key,
        )

        row = await queue._db.fetchone(
            """
            SELECT idempotency_key_digest FROM delivery_idempotency
            WHERE agent_id = ?
            """,
            (queue._agent_id,),
        )
        assert row[0] != raw_key
        assert len(row[0]) == 64

    @pytest.mark.asyncio
    async def test_schema_initialization_is_idempotent(self, real_queue):
        queue, _ = real_queue

        await queue._ensure_tables()

        indexes = await queue._db.fetchall("PRAGMA index_list(delivery_dead_letter)")
        names = {row[1] for row in indexes}
        assert "idx_delivery_dead_letter_original" in names
        assert "idx_delivery_dead_letter_retry_entry" in names

        await queue._ensure_tables()

        row = await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency",
        )
        assert row == (0,)

    @pytest.mark.asyncio
    async def test_delivered_purge_order_uses_covering_index(self, real_queue):
        queue, _ = real_queue

        plan = await queue._db.fetchall(
            """
            EXPLAIN QUERY PLAN
            SELECT id FROM delivery_queue
            WHERE agent_id = ? AND status = ? AND delivered_at < ?
            ORDER BY delivered_at, id
            LIMIT 500
            """,
            (
                queue._agent_id,
                DeliveryStatus.DELIVERED.value,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        details = "\n".join(str(row[-1]) for row in plan)
        assert "idx_delivery_queue_purge" in details
        assert "USE TEMP B-TREE" not in details

    @pytest.mark.asyncio
    async def test_legacy_dead_letter_uses_configured_retry_policy(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = await AsyncDatabase.sqlite(str(tmp_path / "legacy-dead-letter.db"))
        await database.execute(
            """
            CREATE TABLE delivery_dead_letter (
                id TEXT PRIMARY KEY,
                original_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        await database.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-dl",
                "legacy-original",
                "did:test:legacy-dead-letter",
                "email",
                "legacy@example.com",
                '{"body":"hello"}',
                "legacy failure",
                4,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        queue = DeliveryQueue(
            database, "did:test:legacy-dead-letter", max_retries=10
        )
        try:
            await queue._ensure_tables()
            retried = await queue.retry("legacy-original")
            assert retried["success"] is True
            assert await database.fetchone(
                "SELECT max_retries FROM delivery_queue WHERE id = ?",
                (retried["entry_id"],),
            ) == (10,)
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_rolling_legacy_dead_letter_insert_keeps_policy_nullable(
        self, real_queue
    ):
        queue, _ = real_queue
        rolling = DeliveryQueue(queue._db, queue._agent_id, max_retries=10)
        await queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rolling-dl",
                "rolling-original",
                queue._agent_id,
                "email",
                "rolling@example.com",
                '{"body":"hello"}',
                "legacy writer failure",
                4,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        retried = await rolling.retry("rolling-original")

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == (10,)

    @pytest.mark.asyncio
    async def test_legacy_dead_letter_retry_preserves_exact_raw_hash(self, real_queue):
        queue, _ = real_queue
        content_json = '{"1":"first","1":"second"}'
        recipient = "legacy-duplicate-key@example.com"
        await queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-duplicate-key-dl",
                "legacy-duplicate-key-original",
                queue._agent_id,
                "email",
                recipient,
                content_json,
                "legacy writer failure",
                4,
                datetime.now(timezone.utc).isoformat(),
                5,
            ),
        )

        retried = await queue.retry("legacy-duplicate-key-original")

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == (QueueEntry.compute_content_hash(recipient, content_json),)

    @pytest.mark.asyncio
    async def test_v3_upgrade_removes_unscoped_v2_delete_trigger(self, real_queue):
        queue, _ = real_queue
        await queue._db.execute(
            "DROP TRIGGER IF EXISTS trg_delivery_idempotency_compensate"
        )
        await queue._db.execute(
            """
            CREATE TRIGGER trg_delivery_idempotency_delete
            AFTER DELETE ON delivery_idempotency
            BEGIN
                DELETE FROM delivery_queue
                WHERE id = OLD.entry_id AND agent_id = OLD.agent_id;
            END
            """
        )

        await queue._ensure_tables()
        entry_id = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="post-v2-upgrade",
        )
        await queue._db.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        )

        assert await queue._db.fetchone(
            "SELECT id FROM delivery_queue WHERE id = ?",
            (entry_id,),
        ) == (entry_id,)

    @pytest.mark.asyncio
    async def test_schema_has_retention_and_scoped_compensation_indexes(
        self, real_queue
    ):
        queue, _ = real_queue
        indexes = await queue._db.fetchall(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'index' AND tbl_name = 'delivery_idempotency'
            """
        )
        names = {row[0] for row in indexes}
        assert "idx_delivery_idempotency_entry" in names
        assert "idx_delivery_idempotency_previous" in names
        assert "idx_delivery_idempotency_retention" in names

    @pytest.mark.asyncio
    async def test_v3_schema_does_not_rebuild_replay_index(self, real_queue):
        queue, _ = real_queue
        original_execute = queue._db.execute

        with patch.object(queue._db, "execute", wraps=original_execute) as execute:
            await queue._ensure_tables()

        sql = "\n".join(call.args[0] for call in execute.call_args_list)
        assert "DROP INDEX IF EXISTS idx_delivery_idempotency_entry" not in sql

    @pytest.mark.asyncio
    async def test_v2_unique_replay_index_is_replaced_once(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = await AsyncDatabase.sqlite(str(tmp_path / "delivery-v2.db"))
        try:
            await database.execute(
                """
                CREATE TABLE delivery_idempotency (
                    agent_id TEXT NOT NULL,
                    idempotency_key_digest TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (agent_id, idempotency_key_digest)
                )
                """
            )
            await database.execute(
                """
                CREATE UNIQUE INDEX idx_delivery_idempotency_entry
                ON delivery_idempotency(agent_id, entry_id)
                """
            )

            queue = DeliveryQueue(database, "did:test:v2-upgrade")
            await queue._ensure_tables()
            indexes = await database.fetchall(
                "PRAGMA index_list('delivery_idempotency')"
            )
            replay_index = next(
                row for row in indexes if row[1] == "idx_delivery_idempotency_entry"
            )

            assert replay_index[2] == 0
            assert await database.column_exists(
                "delivery_idempotency", "compensating"
            )

            with patch.object(database, "execute", wraps=database.execute) as execute:
                await queue._ensure_tables()
            sql = "\n".join(call.args[0] for call in execute.call_args_list)
            assert "DROP INDEX IF EXISTS idx_delivery_idempotency_entry" not in sql
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_queue_insert_failure_rolls_back_idempotency_claim(self, real_queue):
        queue, _ = real_queue
        await queue._db.execute(
            """
            CREATE TRIGGER reject_delivery_insert
            BEFORE INSERT ON delivery_queue
            BEGIN
                SELECT RAISE(ABORT, 'injected queue insert failure');
            END
            """
        )

        with pytest.raises(TransactionError, match="injected queue insert failure"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"body": "hello"},
                idempotency_key="retry-after-rollback",
            )

        row = await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        )
        assert row == (0,)

        await queue._db.execute("DROP TRIGGER reject_delivery_insert")
        entry_id = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="retry-after-rollback",
        )
        assert entry_id

    @pytest.mark.asyncio
    async def test_caught_nested_insert_failure_does_not_commit_claim(
        self, real_queue
    ):
        queue, _ = real_queue
        await queue._db.execute(
            """
            CREATE TRIGGER reject_nested_delivery_insert
            BEFORE INSERT ON delivery_queue
            BEGIN
                SELECT RAISE(ABORT, 'injected nested queue insert failure');
            END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="injected nested queue insert failure"):
                await queue.enqueue(
                    "email",
                    "person@example.com",
                    {"body": "hello"},
                    idempotency_key="retry-after-nested-failure",
                )

        row = await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        )
        assert row == (0,)

        await queue._db.execute("DROP TRIGGER reject_nested_delivery_insert")
        assert await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="retry-after-nested-failure",
        )

    @pytest.mark.asyncio
    async def test_failure_after_claim_before_queue_insert_compensates_claim(
        self, real_queue
    ):
        queue, _ = real_queue
        original_fetchone = queue._db.fetchone

        async def fail_claim_read(sql, params=()):
            if "SELECT entry_id, payload_digest" in sql:
                raise QueryError("injected claim read failure")
            return await original_fetchone(sql, params)

        async with queue._db.transaction(immediate=True):
            with patch.object(queue._db, "fetchone", side_effect=fail_claim_read):
                with pytest.raises(QueryError, match="injected claim read failure"):
                    await queue.enqueue(
                        "email",
                        "person@example.com",
                        {"body": "hello"},
                        idempotency_key="retry-after-claim-read-failure",
                    )

        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)

    @pytest.mark.asyncio
    async def test_failed_stale_claim_repair_restores_original_claim(
        self, real_queue
    ):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "stale-claim@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "stale-claim-key",
        }
        original_id = await queue.enqueue(**request)
        original_claim = await queue._db.fetchone(
            """
            SELECT payload_digest FROM delivery_idempotency
            WHERE agent_id = ? AND entry_id = ?
            """,
            (queue._agent_id, original_id),
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        await queue._db.execute(
            """
            CREATE TRIGGER reject_stale_claim_queue_insert
            BEFORE INSERT ON delivery_queue
            BEGIN SELECT RAISE(ABORT, 'stale repair failed'); END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="stale repair failed"):
                await queue.enqueue(**request)

        assert await queue._db.fetchone(
            """
            SELECT entry_id, payload_digest, compensating, previous_entry_id
            FROM delivery_idempotency WHERE agent_id = ?
            """,
            (queue._agent_id,),
        ) == (original_id, original_claim[0], 0, None)
        with pytest.raises(DeliveryIdempotencyConflict):
            await queue.enqueue(**{**request, "content": {"body": "changed"}})

    @pytest.mark.asyncio
    async def test_ambiguous_stale_repair_cancel_restores_original_claim(
        self, real_queue
    ):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "ambiguous-stale@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "ambiguous-stale",
        }
        original_id = await queue.enqueue(**request)
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        original_execute = queue._db.execute
        cancelled = False

        async def cancel_after_insert(sql, params=()):
            nonlocal cancelled
            result = await original_execute(sql, params)
            if "INSERT INTO delivery_queue" in sql and not cancelled:
                cancelled = True
                raise asyncio.CancelledError("ambiguous stale insert")
            return result

        async with queue._db.transaction(immediate=True):
            with patch.object(queue._db, "execute", side_effect=cancel_after_insert):
                with pytest.raises(asyncio.CancelledError, match="ambiguous stale"):
                    await queue.enqueue(**request)

        assert await queue._db.fetchone(
            """
            SELECT entry_id, compensating, previous_entry_id
            FROM delivery_idempotency WHERE agent_id = ?
            """,
            (queue._agent_id,),
        ) == (original_id, 0, None)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)

    @pytest.mark.asyncio
    async def test_stale_claim_with_unlinked_rolling_retry_fails_closed(
        self, real_queue
    ):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "rolling-retry@example.com",
            "content": {"z": 1, "a": 2},
            "idempotency_key": "rolling-retry",
        }
        original_id = await queue.enqueue(**request)
        original = await queue._db.fetchone(
            """
            SELECT content_json, content_hash, canonical_content_hash, max_retries
            FROM delivery_queue WHERE id = ?
            """,
            (original_id,),
        )
        claim_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        candidate_time = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()
        await queue._db.execute(
            "UPDATE delivery_idempotency SET created_at = ? WHERE agent_id = ?",
            (claim_time, queue._agent_id),
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, canonical_content_hash, status, attempts,
                 max_retries, next_retry_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                "rolling-retry-candidate",
                queue._agent_id,
                "email",
                request["recipient"],
                original[0],
                original[2],
                None,
                DeliveryStatus.PENDING.value,
                original[3] + 1,
                candidate_time,
                candidate_time,
            ),
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="unlinked"):
            await queue.enqueue(**request)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (1,)

    @pytest.mark.asyncio
    async def test_shared_stale_claims_adopt_one_anchored_replacement(
        self, real_queue
    ):
        queue, _ = real_queue
        request = ("email", "shared-stale@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="shared-stale-one"
        )
        assert await queue.enqueue(
            *request, idempotency_key="shared-stale-two"
        ) == original_id
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        replacement = await queue.enqueue(
            *request, idempotency_key="shared-stale-one"
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, replacement),
        )
        assert await queue.enqueue(
            *request, idempotency_key="shared-stale-two"
        ) == replacement
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (1,)

    @pytest.mark.asyncio
    async def test_stale_claim_adopts_existing_anchored_replacement(self, real_queue):
        queue, _ = real_queue
        request = ("email", "anchored-adoption@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="anchored-adoption-stale"
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        replacement = await queue.enqueue(
            *request, idempotency_key="anchored-adoption-replacement"
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, replacement),
        )
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET previous_entry_id = ?
            WHERE agent_id = ? AND entry_id = ?
            """,
            (original_id, queue._agent_id, replacement),
        )

        assert await queue.enqueue(
            *request, idempotency_key="anchored-adoption-stale"
        ) == replacement
        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_idempotency
            WHERE agent_id = ? AND entry_id = ? AND previous_entry_id = ?
            """,
            (queue._agent_id, replacement, original_id),
        ) == (2,)

    @pytest.mark.asyncio
    async def test_stale_claim_rejects_multiple_anchored_replacements(
        self, real_queue
    ):
        queue, _ = real_queue
        request = ("email", "ambiguous-anchor@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="ambiguous-anchor-stale"
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        first = await queue.enqueue(
            *request, idempotency_key="ambiguous-anchor-first"
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, first),
        )
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET previous_entry_id = ?
            WHERE agent_id = ? AND entry_id = ?
            """,
            (original_id, queue._agent_id, first),
        )
        second = "ambiguous-anchor-second-entry"
        await queue._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, canonical_content_hash, status, attempts,
                 max_retries, next_retry_at, created_at)
            SELECT ?, agent_id, channel_type, recipient, content_json,
                   content_hash, canonical_content_hash, status, attempts,
                   max_retries, next_retry_at, ?
            FROM delivery_queue WHERE id = ?
            """,
            (second, old, first),
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_idempotency
                (agent_id, idempotency_key_digest, entry_id, payload_digest,
                 created_at, previous_entry_id, effective_max_retries,
                 legacy_content_hash)
            SELECT agent_id, ?, ?, payload_digest, ?, ?,
                   effective_max_retries, legacy_content_hash
            FROM delivery_idempotency
            WHERE agent_id = ? AND entry_id = ?
            """,
            (
                "f" * 64,
                second,
                old,
                original_id,
                queue._agent_id,
                first,
            ),
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="multiple anchored"):
            await queue.enqueue(
                *request, idempotency_key="ambiguous-anchor-stale"
            )

    @pytest.mark.asyncio
    async def test_shared_stale_claims_adopt_recent_plain_delivery(self, real_queue):
        queue, _ = real_queue
        request = ("email", "shared-plain@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="shared-plain-one"
        )
        assert await queue.enqueue(
            *request, idempotency_key="shared-plain-two"
        ) == original_id
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        replacement = await queue.enqueue(*request)
        assert await queue.enqueue(
            *request, idempotency_key="shared-plain-one"
        ) == replacement
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, replacement),
        )

        assert await queue.enqueue(
            *request, idempotency_key="shared-plain-two"
        ) == replacement
        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_idempotency
            WHERE agent_id = ? AND entry_id = ? AND previous_entry_id = ?
            """,
            (queue._agent_id, replacement, original_id),
        ) == (2,)

    @pytest.mark.asyncio
    async def test_shared_stale_replacement_dead_letter_is_terminal(self, real_queue):
        queue, _ = real_queue
        request = ("email", "shared-terminal@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="shared-terminal-one"
        )
        assert await queue.enqueue(
            *request, idempotency_key="shared-terminal-two"
        ) == original_id
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        replacement = await queue.enqueue(
            *request, idempotency_key="shared-terminal-one"
        )
        await queue.move_to_dead_letter(replacement, "provider rejected")

        with pytest.raises(DeliveryIdempotencyTerminal, match="dead-letter"):
            await queue.enqueue(*request, idempotency_key="shared-terminal-two")

    @pytest.mark.asyncio
    async def test_shared_stale_replacement_retry_preserves_aliases(self, real_queue):
        queue, _ = real_queue
        request = ("email", "shared-retry@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="shared-retry-one"
        )
        assert await queue.enqueue(
            *request, idempotency_key="shared-retry-two"
        ) == original_id
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        replacement = await queue.enqueue(
            *request, idempotency_key="shared-retry-one"
        )
        await queue.move_to_dead_letter(replacement, "provider rejected")
        retried = await queue.retry(replacement)
        assert retried["success"] is True
        retried_id = retried["entry_id"]
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, retried_id),
        )

        assert await queue.enqueue(
            *request, idempotency_key="shared-retry-two"
        ) == retried_id
        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_idempotency
            WHERE agent_id = ? AND entry_id = ? AND previous_entry_id = ?
            """,
            (queue._agent_id, retried_id, original_id),
        ) == (2,)

    @pytest.mark.asyncio
    async def test_purge_retains_claim_represented_by_live_anchor(self, real_queue):
        queue, _ = real_queue
        request = ("email", "anchored-purge@example.com", {"body": "hello"})
        original_id = await queue.enqueue(
            *request, idempotency_key="anchored-purge-one"
        )
        assert await queue.enqueue(
            *request, idempotency_key="anchored-purge-two"
        ) == original_id
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )
        replacement = await queue.enqueue(
            *request, idempotency_key="anchored-purge-one"
        )
        claims = await queue._db.fetchall(
            """
            SELECT idempotency_key_digest FROM delivery_idempotency
            WHERE agent_id = ? ORDER BY idempotency_key_digest
            """,
            (queue._agent_id,),
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        await queue._db.execute(
            """
            UPDATE delivery_idempotency
            SET entry_id = ?, previous_entry_id = NULL, created_at = ?
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (original_id, old, queue._agent_id, claims[0][0]),
        )

        await queue.purge_delivered(older_than_hours=24)

        assert await queue._db.fetchone(
            """
            SELECT entry_id FROM delivery_idempotency
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (queue._agent_id, claims[0][0]),
        ) == (original_id,)
        assert await queue._db.fetchone(
            "SELECT id FROM delivery_queue WHERE agent_id = ? AND id = ?",
            (queue._agent_id, replacement),
        ) == (replacement,)

    @pytest.mark.asyncio
    async def test_stale_repair_preserves_durable_legacy_hash(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "stale-hash@example.com",
            {"z": 1, "a": 2},
            idempotency_key="stale-hash",
        )
        original_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (original_id,)
        )
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        repaired = await queue.enqueue(
            "email",
            "stale-hash@example.com",
            {"a": 2, "z": 1},
            idempotency_key="stale-hash",
        )

        assert await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (repaired,)
        ) == original_hash

    @pytest.mark.asyncio
    async def test_adopted_key_persists_queue_legacy_hash_for_stale_repair(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email", "adopted-hash@example.com", {"z": 1, "a": 2}
        )
        original_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (original_id,)
        )
        assert await queue.enqueue(
            "email",
            "adopted-hash@example.com",
            {"a": 2, "z": 1},
            idempotency_key="adopted-hash",
        ) == original_id
        assert await queue._db.fetchone(
            "SELECT legacy_content_hash FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == original_hash
        await queue._db.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        )

        repaired = await queue.enqueue(
            "email",
            "adopted-hash@example.com",
            {"a": 2, "z": 1},
            idempotency_key="adopted-hash",
        )

        assert await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (repaired,)
        ) == original_hash

    @pytest.mark.asyncio
    async def test_postgres_failure_relies_on_transaction_rollback(self, queue):
        queue._db.backend_type = "postgres"
        queue._db.nested_transaction_strategy = "savepoint"

        @asynccontextmanager
        async def transaction(*, immediate=False):
            assert immediate
            yield

        queue._db.transaction = transaction
        queue._db.fetchone = AsyncMock(
            side_effect=[None, None, QueryError("injected aborted transaction")]
        )

        with pytest.raises(QueryError, match="injected aborted transaction"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"body": "hello"},
                idempotency_key="postgres-rollback-only",
            )

        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "INSERT INTO delivery_idempotency" in sql
        assert "DELETE FROM delivery_" not in sql

    @pytest.mark.asyncio
    async def test_ambiguous_cancel_compensates_queue_row_and_claim(
        self, real_queue
    ):
        queue, _ = real_queue
        original_execute = queue._db.execute
        inserted_then_cancelled = False

        async def ambiguous_execute(sql, params=()):
            nonlocal inserted_then_cancelled
            result = await original_execute(sql, params)
            if "INSERT INTO delivery_queue" in sql and not inserted_then_cancelled:
                inserted_then_cancelled = True
                raise asyncio.CancelledError
            return result

        async with queue._db.transaction(immediate=True):
            with patch.object(queue._db, "execute", side_effect=ambiguous_execute):
                with pytest.raises(asyncio.CancelledError):
                    await queue.enqueue(
                        "email",
                        "person@example.com",
                        {"body": "hello"},
                        idempotency_key="retry-after-ambiguous-cancel",
                    )

        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)

        assert await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="retry-after-ambiguous-cancel",
        )

    @pytest.mark.asyncio
    async def test_cleanup_cancellation_cannot_commit_partial_compensation(
        self, real_queue
    ):
        queue, _ = real_queue
        original_execute = queue._db.execute
        phase = "queue"

        async def cancel_after_statement(sql, params=()):
            nonlocal phase
            result = await original_execute(sql, params)
            if "INSERT INTO delivery_queue" in sql and phase == "queue":
                phase = "cleanup"
                raise asyncio.CancelledError("queue cancellation")
            if (
                "UPDATE delivery_idempotency SET compensating = 1" in sql
                and phase == "cleanup"
            ):
                phase = "done"
                raise asyncio.CancelledError("cleanup cancellation")
            return result

        async with queue._db.transaction(immediate=True):
            with patch.object(queue._db, "execute", side_effect=cancel_after_statement):
                with pytest.raises(asyncio.CancelledError, match="cleanup cancellation"):
                    await queue.enqueue(
                        "email",
                        "person@example.com",
                        {"body": "hello"},
                        idempotency_key="cancel-during-compensation",
                    )

        assert phase == "done"
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)

        assert await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="cancel-during-compensation",
        )

    @pytest.mark.asyncio
    async def test_dead_letter_replay_is_terminal_until_explicit_retry(
        self, real_queue
    ):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "person@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "dead-letter-stage",
        }
        original_id = await queue.enqueue(**request)
        await queue.move_to_dead_letter(original_id, "provider rejected")

        with pytest.raises(DeliveryIdempotencyTerminal, match="dead-letter"):
            await queue.enqueue(**request)

        retried = await queue.retry(original_id)
        assert retried["success"] is True
        assert retried["entry_id"] != original_id
        assert await queue.enqueue(**request) == retried["entry_id"]
        ledger = await queue._db.fetchone(
            """
            SELECT entry_id FROM delivery_idempotency
            WHERE agent_id = ?
            """,
            (queue._agent_id,),
        )
        assert ledger == (retried["entry_id"],)

    @pytest.mark.asyncio
    async def test_purge_retains_aged_dead_letter_replay_claim(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "retained-terminal@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "retained-terminal",
        }
        original_id = await queue.enqueue(**request)
        await queue.move_to_dead_letter(original_id, "provider rejected")
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await queue._db.execute(
            "UPDATE delivery_idempotency SET created_at = ? WHERE agent_id = ?",
            (old, queue._agent_id),
        )

        assert await queue.purge_delivered(older_than_hours=24) == 0
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (1,)
        with pytest.raises(DeliveryIdempotencyTerminal):
            await queue.enqueue(**request)

    @pytest.mark.asyncio
    async def test_failed_nested_dead_letter_transition_is_fail_closed(
        self, real_queue
    ):
        queue, deliveries = real_queue
        request = {
            "channel_type": "email",
            "recipient": "dead-transition@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "dead-transition",
        }
        original_id = await queue.enqueue(**request)
        await queue._db.execute(
            """
            CREATE TRIGGER reject_dead_letter_queue_delete
            BEFORE DELETE ON delivery_queue
            BEGIN SELECT RAISE(ABORT, 'queue delete failed'); END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="queue delete failed"):
                await queue.move_to_dead_letter(original_id, "provider rejected")

        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ?", (original_id,)
        ) == (1,)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_dead_letter WHERE original_id = ?",
            (original_id,),
        ) == (1,)
        assert await queue.process_pending() == 0
        assert deliveries == []
        counts = await queue.get_status_counts()
        assert counts["pending"] == 0
        assert counts["dead_letter"] == 1
        with pytest.raises(DeliveryIdempotencyTerminal):
            await queue.enqueue(**request)

        await queue._db.execute("DROP TRIGGER reject_dead_letter_queue_delete")
        await queue.move_to_dead_letter(original_id, "provider rejected")
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ?", (original_id,)
        ) == (0,)

    @pytest.mark.asyncio
    async def test_retry_reconciles_live_dead_letter_intermediate_state(
        self, real_queue
    ):
        queue, deliveries = real_queue
        original_id = await queue.enqueue(
            "email",
            "dual-state@example.com",
            {"body": "hello"},
            idempotency_key="dual-state",
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries)
            SELECT ?, id, agent_id, channel_type, recipient, content_json,
                   ?, attempts, created_at, max_retries
            FROM delivery_queue WHERE id = ?
            """,
            ("dual-state-dl", "injected transition failure", original_id),
        )

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert retried["entry_id"] != original_id
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (1,)
        assert await queue.process_pending() == 1
        assert len(deliveries) == 1

    @pytest.mark.asyncio
    async def test_new_key_does_not_adopt_tombstoned_live_row(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "tombstone-dedup@example.com",
            {"body": "hello"},
            idempotency_key="first-key",
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries)
            SELECT ?, id, agent_id, channel_type, recipient, content_json,
                   ?, attempts, created_at, max_retries
            FROM delivery_queue WHERE id = ?
            """,
            ("tombstone-dedup-dl", "injected transition failure", original_id),
        )

        fresh_id = await queue.enqueue(
            "email",
            "tombstone-dedup@example.com",
            {"body": "hello"},
            idempotency_key="second-key",
        )

        assert fresh_id != original_id
        assert await queue.enqueue(
            "email",
            "tombstone-dedup@example.com",
            {"body": "hello"},
            idempotency_key="second-key",
        ) == fresh_id

    @pytest.mark.asyncio
    async def test_dead_letter_retry_preserves_policy_and_legacy_json(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "legacy@example.com",
            {"score": float("nan")},
            max_retries=11,
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == (11,)

    @pytest.mark.asyncio
    async def test_dead_letter_retry_recovers_policy_from_replay_ledger(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "ledger-policy@example.com",
            {"body": "hello"},
            max_retries=11,
            idempotency_key="ledger-policy",
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            "UPDATE delivery_dead_letter SET max_retries = 5 WHERE original_id = ?",
            (original_id,),
        )

        retried = await DeliveryQueue(
            queue._db, queue._agent_id, max_retries=99
        ).retry(original_id)

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == (11,)

    @pytest.mark.asyncio
    async def test_retry_backfills_pre_upgrade_replay_metadata(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "retry-upgrade@example.com",
            "content": {"body": "hello"},
            "max_retries": 11,
            "idempotency_key": "retry-upgrade",
        }
        original_id = await queue.enqueue(**request)
        original_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?",
            (original_id,),
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            UPDATE delivery_idempotency
            SET effective_max_retries = NULL, legacy_content_hash = NULL
            WHERE agent_id = ?
            """,
            (queue._agent_id,),
        )

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert await queue._db.fetchone(
            """
            SELECT effective_max_retries, legacy_content_hash,
                   previous_entry_id
            FROM delivery_idempotency WHERE agent_id = ?
            """,
            (queue._agent_id,),
        ) == (11, original_hash[0], original_id)
        assert await queue.enqueue(**request) == retried["entry_id"]

    @pytest.mark.asyncio
    async def test_dead_letter_retry_rejects_inconsistent_ledger_policies(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "ambiguous-policy@example.com",
            {"body": "hello"},
            idempotency_key="ambiguous-policy-a",
        )
        assert await queue.enqueue(
            "email",
            "ambiguous-policy@example.com",
            {"body": "hello"},
            idempotency_key="ambiguous-policy-b",
        ) == original_id
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET effective_max_retries = 17
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (
                queue._agent_id,
                hashlib.sha256(b"ambiguous-policy-b").hexdigest(),
            ),
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="inconsistent"):
            await queue.retry(original_id)

    @pytest.mark.asyncio
    async def test_dead_letter_retry_rejects_ambiguous_ledger_hashes(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "ambiguous-hash@example.com",
            {"body": "hello"},
            idempotency_key="ambiguous-hash-a",
        )
        assert await queue.enqueue(
            "email",
            "ambiguous-hash@example.com",
            {"body": "hello"},
            idempotency_key="ambiguous-hash-b",
        ) == original_id
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            UPDATE delivery_dead_letter SET legacy_content_hash = NULL
            WHERE original_id = ? AND agent_id = ?
            """,
            (original_id, queue._agent_id),
        )
        await queue._db.execute(
            """
            UPDATE delivery_idempotency SET legacy_content_hash = ?
            WHERE agent_id = ? AND idempotency_key_digest = ?
            """,
            (
                "different-legacy-hash",
                queue._agent_id,
                hashlib.sha256(b"ambiguous-hash-b").hexdigest(),
            ),
        )

        with pytest.raises(
            DeliveryIdempotencyStateError, match="authoritative compatibility hash"
        ):
            await queue.retry(original_id)

    @pytest.mark.asyncio
    async def test_move_to_dead_letter_does_not_duplicate_existing_tombstone(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email", "existing-tombstone@example.com", {"body": "hello"}
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries,
                 legacy_content_hash)
            SELECT ?, id, agent_id, channel_type, recipient, content_json,
                   'prior move', attempts, created_at, max_retries, content_hash
            FROM delivery_queue WHERE id = ? AND agent_id = ?
            """,
            ("existing-tombstone", original_id, queue._agent_id),
        )

        await queue.move_to_dead_letter(original_id, "resumed move")

        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_dead_letter
            WHERE original_id = ? AND agent_id = ?
            """,
            (original_id, queue._agent_id),
        ) == (1,)
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, queue._agent_id),
        ) == (0,)

    @pytest.mark.asyncio
    async def test_dead_letter_retry_preserves_rolling_writer_legacy_hash(
        self, real_queue
    ):
        queue, _ = real_queue
        payload = {"z": 1, "a": 2}
        original_id = await queue.enqueue(
            "email",
            "rolling-hash@example.com",
            payload,
            idempotency_key="rolling-hash",
        )
        original_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (original_id,)
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            UPDATE delivery_dead_letter SET legacy_content_hash = NULL
            WHERE original_id = ?
            """,
            (original_id,),
        )

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == original_hash
        # An older rolling writer hashes its insertion-order JSON and must
        # still adopt the retry row rather than creating a second delivery.
        assert await queue.enqueue(
            "email", "rolling-hash@example.com", payload
        ) == retried["entry_id"]

    @pytest.mark.asyncio
    async def test_dead_letter_authoritative_hash_allows_alias_hashes(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "alias-hash@example.com",
            {"z": 1, "a": 2},
            idempotency_key="alias-hash-one",
        )
        assert await queue.enqueue(
            "email",
            "alias-hash@example.com",
            {"a": 2, "z": 1},
            idempotency_key="alias-hash-two",
        ) == original_id
        original_hash = await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?", (original_id,)
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == original_hash

    @pytest.mark.asyncio
    async def test_concurrent_dead_letter_retry_creates_one_live_row(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "retry-race@example.com",
            {"body": "hello"},
            idempotency_key="retry-race",
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")

        results = await asyncio.gather(*(queue.retry(original_id) for _ in range(8)))

        successes = [result for result in results if result["success"]]
        assert len(successes) == 1
        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_queue
            WHERE agent_id = ? AND recipient = ?
            """,
            (queue._agent_id, "retry-race@example.com"),
        ) == (1,)
        assert await queue.enqueue(
            "email",
            "retry-race@example.com",
            {"body": "hello"},
            idempotency_key="retry-race",
        ) == successes[0]["entry_id"]

    @pytest.mark.asyncio
    async def test_retry_rechecks_dead_letter_after_live_row_race(self, real_queue):
        queue, _ = real_queue
        original_execute = queue._db.execute
        inserted_tombstone = False

        async def inject_winning_dead_letter_move(sql, params=()):
            nonlocal inserted_tombstone
            result = await original_execute(sql, params)
            if (
                "UPDATE delivery_queue SET id = id" in sql
                and result == 0
                and not inserted_tombstone
            ):
                inserted_tombstone = True
                await original_execute(
                    """
                    INSERT INTO delivery_dead_letter
                        (id, original_id, agent_id, channel_type, recipient,
                         content_json, error, attempts, created_at, max_retries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "late-dead-letter",
                        "late-original",
                        queue._agent_id,
                        "email",
                        "late@example.com",
                        '{"body":"hello"}',
                        "concurrent move",
                        5,
                        datetime.now(timezone.utc).isoformat(),
                        7,
                    ),
                )
            return result

        with patch.object(
            queue._db, "execute", side_effect=inject_winning_dead_letter_move
        ):
            retried = await queue.retry("late-original")

        assert inserted_tombstone is True
        assert retried["success"] is True
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (retried["entry_id"],),
        ) == (7,)

    @pytest.mark.asyncio
    async def test_retry_unwraps_public_state_error(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email", "unwrap@example.com", {"body": "hello"}
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        original_execute = queue._db.execute

        async def lose_locked_tombstone(sql, params=()):
            if sql.strip().startswith("DELETE FROM delivery_dead_letter"):
                return 0
            return await original_execute(sql, params)

        with patch.object(queue._db, "execute", side_effect=lose_locked_tombstone):
            with pytest.raises(
                DeliveryIdempotencyStateError,
                match="lost its locked source row",
            ):
                await queue.retry(original_id)

    @pytest.mark.asyncio
    async def test_failed_nested_dead_letter_retry_remains_resumable(
        self, real_queue
    ):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "retry-failure@example.com",
            {"body": "hello"},
            idempotency_key="retry-failure",
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            CREATE TRIGGER reject_dead_letter_retry
            BEFORE INSERT ON delivery_queue
            BEGIN SELECT RAISE(ABORT, 'retry insert failed'); END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="retry insert failed"):
                await queue.retry(original_id)

        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_dead_letter
            WHERE original_id = ? AND agent_id = ?
            """,
            (original_id, queue._agent_id),
        ) == (1,)
        await queue._db.execute("DROP TRIGGER reject_dead_letter_retry")
        retried = await queue.retry(original_id)
        assert retried["success"] is True
        assert await queue.enqueue(
            "email",
            "retry-failure@example.com",
            {"body": "hello"},
            idempotency_key="retry-failure",
        ) == retried["entry_id"]

    @pytest.mark.asyncio
    async def test_partial_retry_candidate_remains_tombstoned(self, real_queue):
        queue, deliveries = real_queue
        request = {
            "channel_type": "email",
            "recipient": "retry-tombstone@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "retry-tombstone",
        }
        original_id = await queue.enqueue(**request)
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await queue._db.execute(
            """
            CREATE TRIGGER reject_dead_letter_retry_delete
            BEFORE DELETE ON delivery_dead_letter
            BEGIN SELECT RAISE(ABORT, 'retry delete failed'); END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="retry delete failed"):
                await queue.retry(original_id)

        retry_row = await queue._db.fetchone(
            """
            SELECT retry_entry_id FROM delivery_dead_letter
            WHERE original_id = ? AND agent_id = ?
            """,
            (original_id, queue._agent_id),
        )
        assert retry_row is not None and retry_row[0] is not None
        candidate_id = retry_row[0]
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ?",
            (candidate_id,),
        ) == (1,)
        assert await queue.get_pending_entries() == []
        assert await queue.process_pending() == 0
        assert deliveries == []
        with pytest.raises(DeliveryIdempotencyTerminal):
            await queue.enqueue(**request)

        await queue._db.execute("DROP TRIGGER reject_dead_letter_retry_delete")
        resumed = await queue.retry(original_id)
        assert resumed["success"] is True
        assert resumed["entry_id"] == candidate_id
        assert await queue.process_pending() == 1
        assert len(deliveries) == 1

    @pytest.mark.asyncio
    async def test_dead_letter_retry_restores_canonical_dedup_hash(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "canonical-retry@example.com",
            {"subject": "hello", "body": "world"},
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")

        retried = await queue.retry(original_id)
        duplicate = await queue.enqueue(
            "email",
            "canonical-retry@example.com",
            {"body": "world", "subject": "hello"},
        )

        assert retried["success"] is True
        assert duplicate == retried["entry_id"]

    @pytest.mark.asyncio
    async def test_delivered_purge_expires_replay_claim(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "person@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "purged-stage",
        }
        original_id = await queue.enqueue(**request)
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await queue._db.execute(
            """
            UPDATE delivery_queue SET status = ?, delivered_at = ?
            WHERE id = ? AND agent_id = ?
            """,
            (
                DeliveryStatus.DELIVERED.value,
                old,
                original_id,
                queue._agent_id,
            ),
        )

        assert await queue.purge_delivered(older_than_hours=24) == 1
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (0,)
        replayed_id = await queue.enqueue(**request)
        assert replayed_id != original_id

    @pytest.mark.asyncio
    async def test_failed_nested_purge_keeps_replay_fail_safe(self, real_queue):
        queue, _ = real_queue
        request = {
            "channel_type": "email",
            "recipient": "purge-failure@example.com",
            "content": {"body": "hello"},
            "idempotency_key": "purge-failure",
        }
        original_id = await queue.enqueue(**request)
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await queue._db.execute(
            """
            UPDATE delivery_queue SET status = ?, delivered_at = ?
            WHERE id = ? AND agent_id = ?
            """,
            (DeliveryStatus.DELIVERED.value, old, original_id, queue._agent_id),
        )
        await queue._db.execute(
            """
            CREATE TRIGGER reject_purge_claim_delete
            BEFORE DELETE ON delivery_idempotency
            BEGIN SELECT RAISE(ABORT, 'purge claim failed'); END
            """
        )

        async with queue._db.transaction(immediate=True):
            with pytest.raises(QueryError, match="purge claim failed"):
                await queue.purge_delivered(older_than_hours=24)

        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE id = ?", (original_id,)
        ) == (0,)
        await queue._db.execute("DROP TRIGGER reject_purge_claim_delete")
        repaired_id = await queue.enqueue(**request)
        assert repaired_id != original_id
        assert await queue._db.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (queue._agent_id,),
        ) == (1,)

    @pytest.mark.asyncio
    async def test_ordinary_ledger_delete_does_not_delete_live_queue(
        self, real_queue
    ):
        queue, _ = real_queue
        entry_id = await queue.enqueue(
            "email",
            "person@example.com",
            {"body": "hello"},
            idempotency_key="prunable-ledger",
        )
        await queue._db.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?",
            (queue._agent_id,),
        )

        assert await queue._db.fetchone(
            "SELECT id FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (entry_id, queue._agent_id),
        ) == (entry_id,)

    @pytest.mark.asyncio
    async def test_keyed_content_requires_stable_json_values(self, real_queue):
        queue, _ = real_queue

        with pytest.raises(ValueError, match="content.when.*datetime"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"when": datetime(2026, 1, 1)},
                idempotency_key="strict-json",
            )
        with pytest.raises(ValueError, match="content.value.*object"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"value": object()},
                idempotency_key="strict-object-json",
            )
        with pytest.raises(ValueError, match="non-string JSON key"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {1: "coerces-to-string"},
                idempotency_key="strict-json-keys",
            )
        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        with pytest.raises(ValueError, match="content.self.*JSON cycle"):
            await queue.enqueue(
                "email",
                "person@example.com",
                cyclic,
                idempotency_key="strict-json-cycle",
            )

    @pytest.mark.asyncio
    async def test_keyed_and_plain_enqueues_share_dedup_identity(self, real_queue):
        queue, _ = real_queue
        content = {"subject": "hello", "body": "world"}

        plain_id = await queue.enqueue("email", "first@example.com", content)
        keyed_after = await queue.enqueue(
            "email",
            "first@example.com",
            {"body": "world", "subject": "hello"},
            idempotency_key="keyed-after-plain",
        )
        assert keyed_after == plain_id
        second_key = await queue.enqueue(
            "email",
            "first@example.com",
            content,
            idempotency_key="second-key-same-dedup-entry",
        )
        assert second_key == plain_id

        keyed_id = await queue.enqueue(
            "email",
            "second@example.com",
            content,
            idempotency_key="keyed-before-plain",
        )
        plain_after = await queue.enqueue(
            "email",
            "second@example.com",
            {"body": "world", "subject": "hello"},
        )
        assert plain_after == keyed_id

    @pytest.mark.asyncio
    async def test_plain_enqueue_hashes_its_single_serialized_snapshot(
        self, real_queue
    ):
        queue, _ = real_queue

        class ChangingString:
            def __init__(self):
                self.calls = 0

            def __str__(self):
                self.calls += 1
                return str(self.calls)

        changing = ChangingString()
        first_id = await queue.enqueue(
            "email", "stateful-string@example.com", {"value": changing}
        )
        second_id = await queue.enqueue(
            "email", "stateful-string@example.com", {"value": "2"}
        )

        assert changing.calls == 1
        assert second_id != first_id
        assert await queue._db.fetchall(
            """
            SELECT content_json FROM delivery_queue
            WHERE agent_id = ? ORDER BY created_at, id
            """,
            (queue._agent_id,),
        ) == [('{"value": "1"}',), ('{"value": "2"}',)]

    @pytest.mark.asyncio
    async def test_keyed_adoption_requires_matching_delivery_semantics(
        self, real_queue
    ):
        queue, _ = real_queue
        content = {"body": "hello"}
        plain_channel = await queue.enqueue(
            "webhook", "channel@example.com", content
        )
        keyed_channel = await queue.enqueue(
            "email",
            "channel@example.com",
            content,
            idempotency_key="channel-key",
        )
        plain_policy = await queue.enqueue(
            "email", "policy@example.com", content
        )
        keyed_policy = await queue.enqueue(
            "email",
            "policy@example.com",
            content,
            max_retries=99,
            idempotency_key="policy-key",
        )

        assert keyed_channel != plain_channel
        assert keyed_policy != plain_policy
        assert await queue._db.fetchone(
            "SELECT channel_type FROM delivery_queue WHERE id = ?",
            (keyed_channel,),
        ) == ("email",)
        assert await queue._db.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (keyed_policy,),
        ) == (99,)

    @pytest.mark.asyncio
    async def test_dedup_lookup_uses_both_content_indexes(self, real_queue):
        queue, _ = real_queue
        original_fetchone = queue._db.fetchone
        captured = {}

        async def capture_dedup_query(sql, params=()):
            if "AS candidates" in sql:
                captured["sql"] = sql
                captured["params"] = params
            return await original_fetchone(sql, params)

        with patch.object(queue._db, "fetchone", side_effect=capture_dedup_query):
            await queue.enqueue(
                "email", "query-plan@example.com", {"body": "hello"}
            )

        plan = await queue._db.fetchall(
            f"EXPLAIN QUERY PLAN {captured['sql']}", captured["params"]
        )
        details = "\n".join(str(column) for row in plan for column in row)
        assert "idx_delivery_queue_dedup" in details
        assert "idx_delivery_queue_canonical_dedup" in details

    @pytest.mark.asyncio
    async def test_status_dead_letter_antijoins_use_identity_indexes(
        self, real_queue
    ):
        queue, _ = real_queue
        original_fetchall = queue._db.fetchall
        captured = {}

        async def capture_status_query(sql, params=()):
            if "GROUP BY status" in sql:
                captured["sql"] = sql
                captured["params"] = params
            return await original_fetchall(sql, params)

        with patch.object(queue._db, "fetchall", side_effect=capture_status_query):
            await queue.get_status_counts()

        normalized = " ".join(captured["sql"].split())
        assert " OR " not in normalized
        plan = await queue._db.fetchall(
            f"EXPLAIN QUERY PLAN {captured['sql']}", captured["params"]
        )
        details = "\n".join(str(column) for row in plan for column in row)
        assert "idx_delivery_dead_letter_original" in details
        assert "idx_delivery_dead_letter_retry" in details

    @pytest.mark.asyncio
    async def test_stale_compatibility_probe_uses_both_content_indexes(
        self, real_queue
    ):
        queue, _ = real_queue
        original_fetchone = queue._db.fetchone
        captured = {}

        async def capture_compatibility_query(sql, params=()):
            if "AS compatible_queue_rows" in sql:
                captured["sql"] = sql
                captured["params"] = params
            return await original_fetchone(sql, params)

        with patch.object(
            queue._db, "fetchone", side_effect=capture_compatibility_query
        ):
            found = await queue._has_unlinked_compatible_queue_row(
                recipient="query-plan@example.com",
                canonical_content_hash="canonical",
                legacy_content_hash="legacy",
                channel_type="email",
                claim_created_at=datetime.now(timezone.utc).isoformat(),
            )

        assert found is False
        normalized = " ".join(captured["sql"].split())
        assert " OR " not in normalized
        assert "UNION ALL" in normalized
        plan = await queue._db.fetchall(
            f"EXPLAIN QUERY PLAN {captured['sql']}", captured["params"]
        )
        details = "\n".join(str(column) for row in plan for column in row)
        assert "idx_delivery_queue_canonical_dedup" in details
        assert "idx_delivery_queue_dedup" in details

    @pytest.mark.asyncio
    async def test_dead_letter_lock_probes_each_identity_without_or(
        self, real_queue
    ):
        queue, _ = real_queue
        original_execute = queue._db.execute

        with patch.object(queue._db, "execute", wraps=original_execute) as execute:
            assert await queue._lock_dead_letter("missing-entry") is None

        updates = [
            " ".join(call.args[0].split())
            for call in execute.call_args_list
            if "UPDATE delivery_dead_letter" in call.args[0]
        ]
        assert len(updates) == 1
        assert all(" OR " not in sql for sql in updates)
        assert "UNION ALL" in updates[0]
        assert "AND id = ?" in updates[0]
        assert "AND original_id = ?" in updates[0]
        assert "AND retry_entry_id = ?" in updates[0]

    @pytest.mark.asyncio
    async def test_pre_upgrade_legacy_hash_still_deduplicates(self, real_queue):
        queue, _ = real_queue
        entry_id = "pre-upgrade-entry"
        content = {"subject": "hello", "body": "world"}
        content_json = json.dumps(content, default=str)
        legacy_hash = QueueEntry.compute_content_hash(
            "legacy-hash@example.com", content_json
        )
        now = datetime.now(timezone.utc).isoformat()
        await queue._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, status, attempts, max_retries,
                 next_retry_at, last_error, created_at, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 5, ?, NULL, ?, NULL)
            """,
            (
                entry_id,
                queue._agent_id,
                "email",
                "legacy-hash@example.com",
                content_json,
                legacy_hash,
                DeliveryStatus.PENDING.value,
                now,
                now,
            ),
        )

        assert await queue.enqueue(
            "email", "legacy-hash@example.com", content
        ) == entry_id

    @pytest.mark.asyncio
    async def test_new_rows_keep_legacy_hash_for_rolling_readers(self, real_queue):
        queue, _ = real_queue
        content = {"subject": "hello", "body": "world"}
        recipient = "rolling-reader@example.com"
        entry_id = await queue.enqueue(
            "email", recipient, content, idempotency_key="rolling-reader"
        )

        row = await queue._db.fetchone(
            """
            SELECT content_hash, canonical_content_hash
            FROM delivery_queue WHERE id = ?
            """,
            (entry_id,),
        )
        assert row == (
            QueueEntry.compute_content_hash(
                recipient, json.dumps(content, default=str)
            ),
            QueueEntry.compute_content_hash(
                recipient, json.dumps(content, sort_keys=True, separators=(",", ":"))
            ),
        )

    @pytest.mark.asyncio
    async def test_schema_upgrade_backfills_canonical_hashes(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "upgrade-hash@example.com",
            {"subject": "hello", "body": "world"},
        )
        await queue._db.execute(
            "UPDATE delivery_queue SET canonical_content_hash = NULL WHERE id = ?",
            (original_id,),
        )

        await queue._ensure_tables()

        assert await queue.enqueue(
            "email",
            "upgrade-hash@example.com",
            {"body": "world", "subject": "hello"},
        ) == original_id

    @pytest.mark.asyncio
    async def test_startup_hash_backfill_is_owner_scoped_and_bounded(
        self, real_queue
    ):
        queue, _ = real_queue
        now = datetime.now(timezone.utc).isoformat()
        await queue._db.execute(
            """
            WITH RECURSIVE sequence(value) AS (
                SELECT 1 UNION ALL SELECT value + 1 FROM sequence WHERE value < 501
            )
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, canonical_content_hash, status, attempts,
                 max_retries, next_retry_at, created_at)
            SELECT 'backfill-' || value, ?, 'email', 'bulk@example.com',
                   '{"body":"hello"}', 'legacy', NULL, 'pending', 0, 5, ?, ?
            FROM sequence
            """,
            (queue._agent_id, now, now),
        )
        await queue._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, canonical_content_hash, status, attempts,
                 max_retries, next_retry_at, created_at)
            VALUES ('other-owner-backfill', 'did:test:other-owner', 'email',
                    'bulk@example.com', '{"body":"hello"}', 'legacy', NULL,
                    'pending', 0, 5, ?, ?)
            """,
            (now, now),
        )

        await queue._ensure_tables()

        assert await queue._db.fetchone(
            """
            SELECT COUNT(*) FROM delivery_queue
            WHERE agent_id = ? AND canonical_content_hash IS NOT NULL
            """,
            (queue._agent_id,),
        ) == (500,)
        assert await queue._db.fetchone(
            """
            SELECT canonical_content_hash FROM delivery_queue
            WHERE id = 'other-owner-backfill'
            """
        ) == (None,)

    @pytest.mark.asyncio
    async def test_enqueue_reconciles_runtime_legacy_writer_hash(self, real_queue):
        queue, _ = real_queue
        original_id = await queue.enqueue(
            "email",
            "runtime-legacy@example.com",
            {"subject": "hello", "body": "world"},
        )
        await queue._db.execute(
            "UPDATE delivery_queue SET canonical_content_hash = NULL WHERE id = ?",
            (original_id,),
        )

        assert await queue.enqueue(
            "email",
            "runtime-legacy@example.com",
            {"body": "world", "subject": "hello"},
        ) == original_id
        assert await queue._db.fetchone(
            "SELECT canonical_content_hash FROM delivery_queue WHERE id = ?",
            (original_id,),
        ) != (None,)

    @pytest.mark.asyncio
    async def test_keyed_and_plain_enqueues_share_sqlite_content_gate(self, tmp_path):
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        path = str(tmp_path / "mixed-enqueue.db")
        first_db = await AsyncDatabase.sqlite(path)
        second_db = await AsyncDatabase.sqlite(path)
        first = DeliveryQueue(first_db, "did:test:mixed-enqueue")
        second = DeliveryQueue(second_db, "did:test:mixed-enqueue")
        await first._ensure_tables()
        try:
            calls = []
            for index in range(20):
                queue = first if index % 2 == 0 else second
                key = f"mixed-{index}" if index % 3 else None
                calls.append(
                    queue.enqueue(
                        "email",
                        "mixed@example.com",
                        {"body": "same"},
                        idempotency_key=key,
                    )
                )
            entry_ids = await asyncio.gather(*calls)

            assert len(set(entry_ids)) == 1
            assert await first_db.fetchone(
                "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
                ("did:test:mixed-enqueue",),
            ) == (1,)
        finally:
            await first_db.close()
            await second_db.close()

    def test_unrelated_exception_context_is_not_reported_as_conflict(self):
        try:
            raise DeliveryIdempotencyConflict("original conflict")
        except DeliveryIdempotencyConflict:
            try:
                raise QueryError("write connection unavailable")
            except QueryError as database_error:
                assert DeliveryQueue._find_idempotency_error(database_error) is None

    @pytest.mark.asyncio
    async def test_unknown_nested_transaction_semantics_fail_closed(self, queue):
        queue._db.nested_transaction_strategy = None

        with pytest.raises(DeliveryIdempotencyStateError, match="does not declare"):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"body": "hello"},
                idempotency_key="unknown-backend",
            )
        queue._db.execute.assert_not_called()


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
        dead_letter_source = (
            "entry-1", "did:test:delivery-agent", "webhook",
            "https://example.com/hook", '{"text": "hello"}', 5,
            datetime.now(timezone.utc).isoformat(),
            5,
            "abc123",
        )
        queue._db.fetchone = AsyncMock(side_effect=[dead_letter_source, None])

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
        queue._db.execute = AsyncMock(side_effect=[0, 1, 1])
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is True
        assert result["status"] == "queued_for_retry"

    @pytest.mark.asyncio
    async def test_retry_already_delivered(self, queue):
        row = _make_queue_row(entry_id="e1", status="delivered")
        queue._db.execute = AsyncMock(side_effect=[0, 1])
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is False
        assert "already delivered" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_retry_in_flight(self, queue):
        row = _make_queue_row(entry_id="e1", status="in_flight")
        queue._db.execute = AsyncMock(side_effect=[0, 1])
        queue._db.fetchone = AsyncMock(return_value=row)

        result = await queue.retry("e1")
        assert result["success"] is False
        assert "in flight" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_retry_from_dead_letter(self, queue):
        dl_row = _make_dead_letter_row(dl_id="dl-1", original_id="e1")
        queue._db.fetchone = AsyncMock(return_value=dl_row)

        result = await queue.retry("e1")
        assert result["success"] is True
        assert result["status"] == "re-enqueued_from_dead_letter"

        # Should have INSERTed into delivery_queue and DELETEd from dead_letter
        execute_calls = queue._db.execute.call_args_list
        assert any("INSERT INTO delivery_queue" in str(c) for c in execute_calls)
        assert any("DELETE FROM delivery_dead_letter" in str(c) for c in execute_calls)

    @pytest.mark.asyncio
    async def test_lost_dead_letter_claim_does_not_insert(self, queue):
        queue._db.execute = AsyncMock(return_value=0)

        result = await queue.retry("e1")

        assert result == {
            "success": False,
            "error": "Entry e1 not found or already retried",
        }
        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "INSERT INTO delivery_queue" not in sql

    @pytest.mark.asyncio
    async def test_retry_not_found(self, queue):
        queue._db.execute = AsyncMock(return_value=0)
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
        row = (
            "e1", "did:test:delivery-agent", "webhook",
            "http://example.com", '{"msg": "hi"}', 5,
            datetime.now(timezone.utc).isoformat(),
            5,
            "abc123",
        )
        queue._db.fetchone = AsyncMock(side_effect=[row, None])

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
        sql = "\n".join(call.args[0] for call in queue._db.execute.call_args_list)
        assert "INSERT INTO delivery_dead_letter" not in sql
        assert "DELETE FROM delivery_queue" not in sql


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
        queue._db.fetchall = AsyncMock(
            return_value=[(f"delivered-{index}",) for index in range(7)]
        )

        purged = await queue.purge_delivered(older_than_hours=24)
        assert purged == 7

        assert "DELETE FROM delivery_queue" in queue._db.fetchall.call_args.args[0]

    @pytest.mark.asyncio
    async def test_purge_nothing_to_purge(self, queue):
        queue._db.fetchall = AsyncMock(return_value=[])

        purged = await queue.purge_delivered()
        assert purged == 0

        # Ledger retention still runs, but no queue row is deleted.
        assert "DELETE FROM delivery_queue" in queue._db.fetchall.call_args.args[0]

    @pytest.mark.asyncio
    async def test_purge_collects_returned_ids_in_bounded_batches(self, queue):
        first = [(f"delivered-{index}",) for index in range(500)]
        queue._db.fetchall = AsyncMock(side_effect=[first, [("delivered-500",)]])

        purged = await queue.purge_delivered(older_than_hours=24)

        assert purged == 501
        assert queue._db.fetchall.call_count == 2
        for call in queue._db.fetchall.call_args_list:
            assert "LIMIT 500" in call.args[0]


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
                mock_create.call_args.args[0].close()
                assert queue._running is True

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, queue):
        import asyncio

        # Create a real asyncio task that we can cancel
        async def noop():
            await asyncio.sleep(3600)

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
        feature._queue.enqueue.assert_awaited_once_with(
            "webhook",
            "http://example.com/hook",
            {"text": "hello"},
            None,
            idempotency_key=None,
        )

    @pytest.mark.asyncio
    async def test_enqueue_via_feature_forwards_idempotency_key(self, feature):
        feature._queue.enqueue = AsyncMock(return_value="existing-entry-id")

        entry_id = await feature.enqueue_message(
            channel_type="email",
            recipient="person@example.com",
            content={"body": "hello"},
            max_retries=8,
            idempotency_key="workflow-stage-key",
        )

        assert entry_id == "existing-entry-id"
        feature._queue.enqueue.assert_awaited_once_with(
            "email",
            "person@example.com",
            {"body": "hello"},
            8,
            idempotency_key="workflow-stage-key",
        )

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
# DeliveryQueue - in-flight reclamation on start (F163)
# =========================================================================


class TestQueueReclaimInFlight:
    """A crash/restart leaves rows IN_FLIGHT forever; start() must reclaim them."""

    @pytest_asyncio.fixture
    async def real_queue(self, tmp_path):
        """DeliveryQueue backed by a real (file) SQLite database."""
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        database = await AsyncDatabase.sqlite(str(tmp_path / "delivery.db"))
        q = DeliveryQueue(database, "did:test:reclaim-agent")
        await q._ensure_tables()
        yield q
        await q.stop()
        await database.close()

    async def _insert(self, q, entry_id, status, attempts):
        now = datetime.now(timezone.utc).isoformat()
        await q._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json, content_hash,
                 status, attempts, max_retries, next_retry_at, last_error,
                 created_at, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 5, ?, NULL, ?, NULL)
            """,
            (entry_id, q._agent_id, "webhook", "https://example.com/hook",
             '{"text": "hi"}', f"hash-{entry_id}", status, attempts, now, now),
        )

    async def _status_attempts(self, q, entry_id):
        row = await q._db.fetchone(
            "SELECT status, attempts FROM delivery_queue WHERE id = ?",
            (entry_id,),
        )
        return row[0], row[1]

    @pytest.mark.asyncio
    async def test_in_flight_row_reclaimed_preserving_attempts(self, real_queue):
        await self._insert(real_queue, "stuck", DeliveryStatus.IN_FLIGHT.value, 2)

        reclaimed = await real_queue._reclaim_in_flight()

        assert reclaimed == 1
        status, attempts = await self._status_attempts(real_queue, "stuck")
        assert status == DeliveryStatus.PENDING.value
        assert attempts == 2  # attempts preserved

    @pytest.mark.asyncio
    async def test_tombstoned_in_flight_row_is_not_reclaimed(self, real_queue):
        await self._insert(real_queue, "tombstoned", DeliveryStatus.IN_FLIGHT.value, 2)
        now = datetime.now(timezone.utc).isoformat()
        await real_queue._db.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tombstoned-dl",
                "tombstoned",
                real_queue._agent_id,
                "webhook",
                "https://example.com/hook",
                '{"text":"hi"}',
                "interrupted move",
                2,
                now,
                5,
            ),
        )

        assert await real_queue._reclaim_in_flight() == 0
        assert (await self._status_attempts(real_queue, "tombstoned"))[0] == (
            DeliveryStatus.IN_FLIGHT.value
        )

    @pytest.mark.asyncio
    async def test_start_reclaims_in_flight(self, real_queue):
        await self._insert(real_queue, "stuck", DeliveryStatus.IN_FLIGHT.value, 1)

        # start() launches the worker; it must reclaim before/at launch.
        await real_queue.start()

        status, attempts = await self._status_attempts(real_queue, "stuck")
        assert status == DeliveryStatus.PENDING.value
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_reclaim_leaves_other_statuses_untouched(self, real_queue):
        await self._insert(real_queue, "pend", DeliveryStatus.PENDING.value, 0)
        await self._insert(real_queue, "done", DeliveryStatus.DELIVERED.value, 3)
        await self._insert(real_queue, "flight", DeliveryStatus.IN_FLIGHT.value, 1)

        reclaimed = await real_queue._reclaim_in_flight()

        assert reclaimed == 1  # only the in_flight row
        assert (await self._status_attempts(real_queue, "pend"))[0] == DeliveryStatus.PENDING.value
        assert (await self._status_attempts(real_queue, "done"))[0] == DeliveryStatus.DELIVERED.value
        assert (await self._status_attempts(real_queue, "flight"))[0] == DeliveryStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_reclaim_scoped_to_agent(self, real_queue):
        # A row owned by a different agent must not be touched.
        await self._insert(real_queue, "mine", DeliveryStatus.IN_FLIGHT.value, 0)
        now = datetime.now(timezone.utc).isoformat()
        await real_queue._db.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json, content_hash,
                 status, attempts, max_retries, next_retry_at, last_error,
                 created_at, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 5, ?, NULL, ?, NULL)
            """,
            ("theirs", "did:test:other-agent", "webhook", "https://example.com/hook",
             '{"text": "hi"}', "hash-theirs", DeliveryStatus.IN_FLIGHT.value, now, now),
        )

        reclaimed = await real_queue._reclaim_in_flight()

        assert reclaimed == 1
        assert (await self._status_attempts(real_queue, "theirs"))[0] == DeliveryStatus.IN_FLIGHT.value

    @pytest.mark.asyncio
    async def test_reclaim_noop_when_nothing_in_flight(self, real_queue):
        await self._insert(real_queue, "pend", DeliveryStatus.PENDING.value, 0)
        reclaimed = await real_queue._reclaim_in_flight()
        assert reclaimed == 0


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
