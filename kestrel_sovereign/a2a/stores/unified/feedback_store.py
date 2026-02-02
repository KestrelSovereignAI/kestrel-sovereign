"""
Unified FeedbackStore - Backend-Agnostic Agent Feedback.

Enables agents to:
- Log self-observed issues (bugs, suggestions)
- Collect user feedback
- Track feedback resolution
- Analyze patterns in feedback

Works with both SQLite and PostgreSQL backends.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class FeedbackCategory(str, Enum):
    """Category of feedback."""

    BUG = "bug"
    IMPROVEMENT = "improvement"
    SUGGESTION = "suggestion"
    PRAISE = "praise"
    CONFUSION = "confusion"  # User was confused
    ERROR = "error"  # System error
    OTHER = "other"


class FeedbackSeverity(str, Enum):
    """Severity level of feedback."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(str, Enum):
    """Status of feedback item."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"


class FeedbackSource(str, Enum):
    """Source of the feedback."""

    AGENT = "agent"  # Agent self-diagnosed
    USER = "user"  # User submitted
    SYSTEM = "system"  # Automated detection


class FeedbackEntry(BaseModel):
    """A feedback entry from user or agent self-diagnosis."""

    feedback_id: str
    agent_name: str
    session_id: Optional[str]
    source: FeedbackSource
    category: FeedbackCategory
    severity: FeedbackSeverity
    status: FeedbackStatus
    title: str
    description: str
    context: dict[str, Any]  # Conversation context, error details, etc.
    resolution: Optional[str]
    created_at: Any  # datetime
    updated_at: Any  # datetime
    resolved_at: Optional[Any]  # datetime
    metadata: dict[str, Any]


class FeedbackStore(UnifiedStoreBase):
    """
    Backend-agnostic feedback store.

    Replaces both SQLiteFeedbackStore and PostgresFeedbackStore
    with a single implementation that works with any DatabaseBackend.
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize feedback store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()

        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_feedback (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                session_id TEXT,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                context {json_type} DEFAULT '{{}}',
                resolution TEXT,
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                resolved_at {ts_type},
                metadata {json_type} DEFAULT '{{}}'
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_agent ON a2a_feedback(agent_name, status)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_status ON a2a_feedback(status, severity)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_category ON a2a_feedback(category)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON a2a_feedback(created_at)"
        )

        logger.info(f"FeedbackStore initialized ({self._backend.backend_type})")

    async def submit_feedback(
        self,
        agent_name: str,
        source: FeedbackSource,
        category: FeedbackCategory,
        severity: FeedbackSeverity,
        title: str,
        description: str,
        session_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Submit feedback. Returns feedback_id."""
        feedback_id = generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_feedback
            (id, agent_name, session_id, source, category, severity, title, description, context, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                agent_name,
                session_id,
                source.value,
                category.value,
                severity.value,
                title,
                description,
                json_dumps(context or {}),
                now,
                now,
                json_dumps(metadata or {}),
            ),
        )

        logger.info(f"Feedback submitted: {feedback_id} [{category.value}] {title}")
        return feedback_id

    async def update_status(
        self,
        feedback_id: str,
        status: FeedbackStatus,
        resolution: Optional[str] = None,
    ) -> None:
        """Update feedback status."""
        is_resolved = status in (
            FeedbackStatus.RESOLVED,
            FeedbackStatus.WONT_FIX,
            FeedbackStatus.DUPLICATE,
        )

        if is_resolved:
            await self._backend.execute(
                f"""
                UPDATE a2a_feedback
                SET status = ?, resolution = ?, updated_at = {self.now_sql()}, resolved_at = {self.now_sql()}
                WHERE id = ?
                """,
                (status.value, resolution, feedback_id),
            )
        else:
            await self._backend.execute(
                f"""
                UPDATE a2a_feedback
                SET status = ?, resolution = ?, updated_at = {self.now_sql()}
                WHERE id = ?
                """,
                (status.value, resolution, feedback_id),
            )

    async def get_feedback(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """Get a specific feedback entry by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_feedback WHERE id = ?",
            (feedback_id,),
        )
        if not row:
            return None
        return self._row_to_entry(row)

    async def query_feedback(
        self,
        agent_name: Optional[str] = None,
        source: Optional[FeedbackSource] = None,
        category: Optional[FeedbackCategory] = None,
        severity: Optional[FeedbackSeverity] = None,
        status: Optional[FeedbackStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[FeedbackEntry]:
        """Query feedback with filters."""
        conditions = []
        params: list[Any] = []

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if source:
            conditions.append("source = ?")
            params.append(source.value)
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if since:
            conditions.append("created_at >= ?")
            params.append(self.to_timestamp_param(since))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_feedback
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_entry(row) for row in rows]

    async def get_open_feedback(
        self,
        agent_name: Optional[str] = None,
        min_severity: FeedbackSeverity = FeedbackSeverity.LOW,
    ) -> list[FeedbackEntry]:
        """Get all open feedback, optionally filtered by agent and minimum severity."""
        severity_order = {
            FeedbackSeverity.LOW: 0,
            FeedbackSeverity.MEDIUM: 1,
            FeedbackSeverity.HIGH: 2,
            FeedbackSeverity.CRITICAL: 3,
        }
        min_level = severity_order[min_severity]
        valid_severities = [s.value for s, level in severity_order.items() if level >= min_level]

        # Build IN clause with proper placeholders
        severity_placeholders = ", ".join("?" for _ in valid_severities)

        conditions = [
            "status IN ('open', 'acknowledged', 'in_progress')",
            f"severity IN ({severity_placeholders})",
        ]
        params: list[Any] = list(valid_severities)

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_feedback
            WHERE {' AND '.join(conditions)}
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END,
                created_at DESC
            """,
            tuple(params),
        )
        return [self._row_to_entry(row) for row in rows]

    async def get_feedback_stats(
        self,
        agent_name: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get feedback statistics (counts by category, severity, status)."""
        conditions = []
        params: list[Any] = []

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if since:
            conditions.append("created_at >= ?")
            params.append(self.to_timestamp_param(since))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Total count
        total = await self._backend.fetch_val(
            f"SELECT COUNT(*) FROM a2a_feedback {where}",
            tuple(params),
        )

        # By category
        cat_rows = await self._backend.fetch_all(
            f"""
            SELECT category, COUNT(*) as count
            FROM a2a_feedback {where}
            GROUP BY category
            """,
            tuple(params),
        )
        by_category = {row[0]: row[1] for row in cat_rows}

        # By severity
        sev_rows = await self._backend.fetch_all(
            f"""
            SELECT severity, COUNT(*) as count
            FROM a2a_feedback {where}
            GROUP BY severity
            """,
            tuple(params),
        )
        by_severity = {row[0]: row[1] for row in sev_rows}

        # By status
        status_rows = await self._backend.fetch_all(
            f"""
            SELECT status, COUNT(*) as count
            FROM a2a_feedback {where}
            GROUP BY status
            """,
            tuple(params),
        )
        by_status = {row[0]: row[1] for row in status_rows}

        # By source
        source_rows = await self._backend.fetch_all(
            f"""
            SELECT source, COUNT(*) as count
            FROM a2a_feedback {where}
            GROUP BY source
            """,
            tuple(params),
        )
        by_source = {row[0]: row[1] for row in source_rows}

        return {
            "total": total or 0,
            "by_category": by_category,
            "by_severity": by_severity,
            "by_status": by_status,
            "by_source": by_source,
        }

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback entry. Returns True if deleted."""
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_feedback WHERE id = ?",
            (feedback_id,),
        )
        return rows_affected > 0

    async def prune_resolved(self, older_than_days: int = 90) -> int:
        """Delete old resolved feedback. Returns count deleted."""
        interval = self.interval_days(older_than_days)
        rows_affected = await self._backend.execute(
            f"""
            DELETE FROM a2a_feedback
            WHERE status IN ('resolved', 'wont_fix', 'duplicate')
              AND resolved_at < {interval}
            """
        )
        return rows_affected

    def _row_to_entry(self, row: tuple) -> FeedbackEntry:
        """
        Convert database row to FeedbackEntry object.

        Row columns (in order):
        0: id, 1: agent_name, 2: session_id, 3: source, 4: category,
        5: severity, 6: status, 7: title, 8: description, 9: context,
        10: resolution, 11: created_at, 12: updated_at, 13: resolved_at, 14: metadata
        """
        return FeedbackEntry(
            feedback_id=row[0],
            agent_name=row[1],
            session_id=row[2],
            source=FeedbackSource(row[3]),
            category=FeedbackCategory(row[4]),
            severity=FeedbackSeverity(row[5]),
            status=FeedbackStatus(row[6]),
            title=row[7],
            description=row[8],
            context=json_loads(row[9]) if row[9] else {},
            resolution=row[10],
            created_at=self.from_timestamp_field(row[11]),
            updated_at=self.from_timestamp_field(row[12]),
            resolved_at=self.from_timestamp_field(row[13]) if row[13] else None,
            metadata=json_loads(row[14]) if row[14] else {},
        )
