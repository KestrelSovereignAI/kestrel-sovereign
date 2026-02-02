"""
Unified ObservabilityStore - Backend-Agnostic Telemetry.

Provides production debugging capabilities:
- Tool call logging with duration tracking
- Error logging with context
- Custom metrics
- Event querying for debugging

Works with both SQLite and PostgreSQL backends.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)


class ObservabilityEvent(BaseModel):
    """A single observability event."""

    event_id: str
    timestamp: Any  # datetime
    agent_name: str
    session_id: Optional[str]
    event_type: str  # 'tool_call', 'tool_response', 'agent_response', 'llm_call', 'error', 'metric'
    tool_name: Optional[str]
    duration_ms: Optional[int]
    success: bool
    error_message: Optional[str]
    metadata: dict[str, Any]


class LLMCallEvent(BaseModel):
    """Detailed LLM call event for observability."""

    event_id: str
    timestamp: Any
    session_id: Optional[str]
    companion_id: Optional[str]
    user_id: Optional[str]
    provider: str
    model: str
    system_prompt_preview: Optional[str]  # First 200 chars
    user_prompt_preview: Optional[str]  # First 200 chars
    response_preview: Optional[str]  # First 200 chars
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    duration_ms: int
    success: bool
    error_message: Optional[str]
    tool_calls: Optional[list[dict]]  # If tools were called
    metadata: dict[str, Any]


class ObservabilityStore(UnifiedStoreBase):
    """
    Backend-agnostic observability store.

    Replaces both SQLiteObservabilityStore and PostgresObservabilityStore
    with a single implementation that works with any DatabaseBackend.
    """

    def __init__(self, backend: DatabaseBackend):
        """
        Initialize observability store with database backend.

        Args:
            backend: DatabaseBackend instance (SQLite or PostgreSQL)
        """
        super().__init__(backend)

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        bool_type = self.boolean_type()

        # Original observability table for tool calls, metrics, etc.
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_observability (
                id TEXT PRIMARY KEY,
                timestamp {ts_type} {ts_default},
                agent_name TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                tool_name TEXT,
                duration_ms INTEGER,
                success {bool_type} DEFAULT {self.to_bool_param(True)},
                error_message TEXT,
                metadata {json_type} DEFAULT '{{}}'
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_agent_type ON a2a_observability(agent_name, event_type)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON a2a_observability(timestamp)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_session ON a2a_observability(session_id)"
        )

        # New LLM calls table for detailed LLM observability
        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_llm_calls (
                id TEXT PRIMARY KEY,
                timestamp {ts_type} {ts_default},
                session_id TEXT,
                companion_id TEXT,
                user_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                system_prompt_preview TEXT,
                user_prompt_preview TEXT,
                response_preview TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                duration_ms INTEGER NOT NULL,
                success {bool_type} DEFAULT {self.to_bool_param(True)},
                error_message TEXT,
                tool_calls {json_type},
                metadata {json_type} DEFAULT '{{}}'
            )
        """)

        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON a2a_llm_calls(timestamp)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON a2a_llm_calls(session_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_companion ON a2a_llm_calls(companion_id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON a2a_llm_calls(model)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_provider ON a2a_llm_calls(provider)"
        )

        logger.info(f"ObservabilityStore initialized ({self._backend.backend_type})")

    async def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log start of tool call. Returns event_id for timing."""
        event_id = generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_observability
            (id, timestamp, agent_name, session_id, event_type, tool_name, metadata)
            VALUES (?, ?, ?, ?, 'tool_call', ?, ?)
            """,
            (
                event_id,
                now,
                agent_name,
                session_id,
                tool_name,
                json_dumps(metadata or {}),
            ),
        )

        return event_id

    async def log_tool_response(
        self,
        event_id: str,
        success: bool,
        duration_ms: int,
        error_message: Optional[str] = None,
    ) -> None:
        """Log completion of tool call with timing."""
        await self._backend.execute(
            """
            UPDATE a2a_observability
            SET success = ?, duration_ms = ?, error_message = ?,
                event_type = 'tool_response'
            WHERE id = ?
            """,
            (self.to_bool_param(success), duration_ms, error_message, event_id),
        )

    async def log_agent_response(
        self,
        agent_name: str,
        duration_ms: int,
        success: bool = True,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an agent response. Returns event_id."""
        event_id = generate_id()
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_observability
            (id, timestamp, agent_name, session_id, event_type, duration_ms, success, metadata)
            VALUES (?, ?, ?, ?, 'agent_response', ?, ?, ?)
            """,
            (
                event_id,
                now,
                agent_name,
                session_id,
                duration_ms,
                self.to_bool_param(success),
                json_dumps(metadata or {}),
            ),
        )

        return event_id

    async def log_error(
        self,
        agent_name: str,
        error_type: str,
        error_message: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an error event. Returns event_id."""
        event_id = generate_id()
        meta = metadata or {}
        meta["error_type"] = error_type
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_observability
            (id, timestamp, agent_name, session_id, event_type, success, error_message, metadata)
            VALUES (?, ?, ?, ?, 'error', ?, ?, ?)
            """,
            (
                event_id,
                now,
                agent_name,
                session_id,
                self.to_bool_param(False),
                error_message,
                json_dumps(meta),
            ),
        )

        return event_id

    async def log_metric(
        self,
        agent_name: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a metric value. Returns event_id."""
        event_id = generate_id()
        meta = metadata or {}
        meta["metric_name"] = metric_name
        meta["metric_value"] = metric_value
        now = self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_observability
            (id, timestamp, agent_name, event_type, metadata)
            VALUES (?, ?, ?, 'metric', ?)
            """,
            (event_id, now, agent_name, json_dumps(meta)),
        )

        return event_id

    async def query_events(
        self,
        agent_name: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[ObservabilityEvent]:
        """Query observability events with filters."""
        conditions = []
        params: list[Any] = []

        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(self.to_timestamp_param(since))
        if until:
            conditions.append("timestamp <= ?")
            params.append(self.to_timestamp_param(until))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_observability
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_event(row) for row in rows]

    async def get_event(self, event_id: str) -> Optional[ObservabilityEvent]:
        """Get a specific event by ID."""
        row = await self._backend.fetch_one(
            "SELECT * FROM a2a_observability WHERE id = ?",
            (event_id,),
        )
        if not row:
            return None
        return self._row_to_event(row)

    async def prune_old_events(self, older_than_days: int = 30) -> int:
        """Delete old events. Returns count deleted."""
        interval = self.interval_days(older_than_days)
        rows_affected = await self._backend.execute(
            f"DELETE FROM a2a_observability WHERE timestamp < {interval}"
        )
        return rows_affected

    def _row_to_event(self, row: tuple) -> ObservabilityEvent:
        """
        Convert database row to ObservabilityEvent object.

        Row columns (in order):
        0: id, 1: timestamp, 2: agent_name, 3: session_id, 4: event_type,
        5: tool_name, 6: duration_ms, 7: success, 8: error_message, 9: metadata
        """
        return ObservabilityEvent(
            event_id=row[0],
            timestamp=self.from_timestamp_field(row[1]),
            agent_name=row[2],
            session_id=row[3],
            event_type=row[4],
            tool_name=row[5],
            duration_ms=row[6],
            success=self.from_bool_field(row[7]),
            error_message=row[8],
            metadata=json_loads(row[9]) if row[9] else {},
        )

    # ==========================================================================
    # LLM Call Observability (A2A-compatible)
    # ==========================================================================

    async def log_llm_call(
        self,
        provider: str,
        model: str,
        duration_ms: int,
        success: bool = True,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        response: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        error_message: Optional[str] = None,
        tool_calls: Optional[list[dict]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Log an LLM call with full details for observability.

        This is the key method for debugging model selection issues.
        Every LLM call should be logged here.

        Args:
            provider: Provider name (e.g., "openai", "ollama", "xai")
            model: Model name (e.g., "gpt-5.1", "llama3.2")
            duration_ms: Call duration in milliseconds
            success: Whether the call succeeded
            session_id: Optional A2A session ID
            companion_id: Optional companion UUID
            user_id: Optional user UUID
            system_prompt: Full system prompt (stored as preview)
            user_prompt: Full user prompt (stored as preview)
            response: Full response (stored as preview)
            input_tokens: Token count for input
            output_tokens: Token count for output
            error_message: Error message if failed
            tool_calls: List of tool calls if any
            metadata: Additional metadata dict

        Returns:
            event_id for the logged call
        """
        event_id = generate_id()
        now = self.now_utc_param()

        # Truncate prompts/response to preview (first 500 chars)
        preview_len = 500
        system_preview = system_prompt[:preview_len] if system_prompt else None
        user_preview = user_prompt[:preview_len] if user_prompt else None
        response_preview = response[:preview_len] if response else None

        await self._backend.execute(
            """
            INSERT INTO a2a_llm_calls
            (id, timestamp, session_id, companion_id, user_id, provider, model,
             system_prompt_preview, user_prompt_preview, response_preview,
             input_tokens, output_tokens, duration_ms, success, error_message,
             tool_calls, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                now,
                session_id,
                companion_id,
                user_id,
                provider,
                model,
                system_preview,
                user_preview,
                response_preview,
                input_tokens,
                output_tokens,
                duration_ms,
                self.to_bool_param(success),
                error_message,
                json_dumps(tool_calls) if tool_calls else None,
                json_dumps(metadata or {}),
            ),
        )

        logger.debug(
            f"Logged LLM call: {provider}/{model} "
            f"(duration: {duration_ms}ms, success: {success})"
        )
        return event_id

    async def query_llm_calls(
        self,
        session_id: Optional[str] = None,
        companion_id: Optional[str] = None,
        user_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        success: Optional[bool] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[LLMCallEvent]:
        """
        Query LLM call events with filters.

        Args:
            session_id: Filter by session
            companion_id: Filter by companion
            user_id: Filter by user
            provider: Filter by provider
            model: Filter by model
            success: Filter by success status
            since: Start timestamp
            until: End timestamp
            limit: Max results to return

        Returns:
            List of LLMCallEvent objects
        """
        conditions = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if companion_id:
            conditions.append("companion_id = ?")
            params.append(companion_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if success is not None:
            conditions.append("success = ?")
            params.append(self.to_bool_param(success))
        if since:
            conditions.append("timestamp >= ?")
            params.append(self.to_timestamp_param(since))
        if until:
            conditions.append("timestamp <= ?")
            params.append(self.to_timestamp_param(until))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._backend.fetch_all(
            f"""
            SELECT * FROM a2a_llm_calls
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_llm_call(row) for row in rows]

    async def get_llm_stats(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Get aggregated LLM usage statistics.

        Returns:
            Dictionary with stats:
            - total_calls
            - success_rate
            - avg_duration_ms
            - total_input_tokens
            - total_output_tokens
            - calls_by_provider
            - calls_by_model
        """
        conditions = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(self.to_timestamp_param(since))
        if until:
            conditions.append("timestamp <= ?")
            params.append(self.to_timestamp_param(until))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get overall stats
        row = await self._backend.fetch_one(
            f"""
            SELECT
                COUNT(*) as total_calls,
                AVG(duration_ms) as avg_duration_ms,
                SUM(CASE WHEN success = {self.to_bool_param(True)} THEN 1 ELSE 0 END) as success_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens
            FROM a2a_llm_calls
            {where}
            """,
            tuple(params),
        )

        total_calls = row[0] if row else 0
        success_count = row[2] if row else 0
        success_rate = (success_count / total_calls * 100) if total_calls > 0 else 0

        # Get calls by provider
        provider_rows = await self._backend.fetch_all(
            f"""
            SELECT provider, COUNT(*) as count
            FROM a2a_llm_calls
            {where}
            GROUP BY provider
            ORDER BY count DESC
            """,
            tuple(params),
        )
        calls_by_provider = {r[0]: r[1] for r in provider_rows}

        # Get calls by model
        model_rows = await self._backend.fetch_all(
            f"""
            SELECT model, COUNT(*) as count
            FROM a2a_llm_calls
            {where}
            GROUP BY model
            ORDER BY count DESC
            """,
            tuple(params),
        )
        calls_by_model = {r[0]: r[1] for r in model_rows}

        return {
            "total_calls": total_calls,
            "success_rate": round(success_rate, 2),
            "avg_duration_ms": round(row[1], 2) if row and row[1] else 0,
            "total_input_tokens": row[3] if row else 0,
            "total_output_tokens": row[4] if row else 0,
            "calls_by_provider": calls_by_provider,
            "calls_by_model": calls_by_model,
        }

    def _row_to_llm_call(self, row: tuple) -> LLMCallEvent:
        """
        Convert database row to LLMCallEvent object.

        Row columns (in order):
        0: id, 1: timestamp, 2: session_id, 3: companion_id, 4: user_id,
        5: provider, 6: model, 7: system_prompt_preview, 8: user_prompt_preview,
        9: response_preview, 10: input_tokens, 11: output_tokens, 12: duration_ms,
        13: success, 14: error_message, 15: tool_calls, 16: metadata
        """
        return LLMCallEvent(
            event_id=row[0],
            timestamp=self.from_timestamp_field(row[1]),
            session_id=row[2],
            companion_id=row[3],
            user_id=row[4],
            provider=row[5],
            model=row[6],
            system_prompt_preview=row[7],
            user_prompt_preview=row[8],
            response_preview=row[9],
            input_tokens=row[10],
            output_tokens=row[11],
            duration_ms=row[12],
            success=self.from_bool_field(row[13]),
            error_message=row[14],
            tool_calls=json_loads(row[15]) if row[15] else None,
            metadata=json_loads(row[16]) if row[16] else {},
        )
