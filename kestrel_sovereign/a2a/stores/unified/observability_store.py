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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel

from kestrel_sovereign.a2a.stores.base import generate_id, json_dumps, json_loads
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.storage.db.interface import DatabaseBackend

logger = logging.getLogger(__name__)

MAX_TOOL_ARGS_JSON_BYTES = 2048
MAX_TOOL_ERROR_MESSAGE_CHARS = 1024

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "bearer",
    "client_secret",
    "cookie",
    "password",
    "private_key",
    "secret",
    "token",
}

# Placeholder written into content-bearing JSON columns (args/metadata) when
# the active privacy config forbids persisting user content (EPHEMERAL /
# ISOLATED). The row itself still persists so counts/latency/status keep
# metering — only the payload is elided (F076).
_CONTENT_GATED_MARKER = "_privacy_gated"

# Lazy import of the PII redactor to avoid importing the privacy feature at
# module load time (mirrors storage.privacy_wrapper.get_anonymize_text).
_anonymize_text = None


def _get_anonymize_text():
    """Lazy-load the anonymize_text function to avoid circular imports."""
    global _anonymize_text
    if _anonymize_text is None:
        from kestrel_sovereign.features.privacy.pii_detector import anonymize_text

        _anonymize_text = anonymize_text
    return _anonymize_text


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
    agent_did: Optional[str] = None
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


@dataclass(frozen=True)
class ToolDispatchEntry:
    """Structured row for a model-requested tool/subagent dispatch."""

    agent_did: str
    session_id: Optional[str]
    turn_id: str
    tool_name: str
    adapter: str
    args_redacted: Any
    result_status: str
    error_class: Optional[str]
    error_message: Optional[str]
    latency_ms: int
    result_size_bytes: Optional[int]


ToolCallLogEntry = ToolDispatchEntry


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
        # Privacy gate for content-bearing payloads (F076). The agent binds a
        # callable returning the live ``PrivacyConfig`` via
        # :meth:`set_privacy_config_provider`; captured by reference so
        # mid-session privacy-mode flips are picked up automatically. ``None``
        # (tests, standalone use) means "no gating" — payloads persist as-is,
        # preserving the legacy behaviour.
        self._privacy_config_provider: Optional[Callable[[], Any]] = None

    def set_privacy_config_provider(
        self, provider: Optional[Callable[[], Any]]
    ) -> None:
        """Bind a callable returning the live ``PrivacyConfig`` (F076).

        Mirrors ``LLMService.set_force_local_only_provider`` (#1492): the
        observability sink is a layer boundary that persists tool-call
        arguments and metadata, so it must honour the agent's privacy mode
        just like ``PrivacyEnforcingStorage`` does for conversation content.
        """
        self._privacy_config_provider = provider

    def _current_privacy_config(self) -> Optional[Any]:
        """Read the live privacy config, tolerating a missing/faulty provider."""
        if self._privacy_config_provider is None:
            return None
        try:
            return self._privacy_config_provider()
        except Exception as exc:  # noqa: BLE001 - never break logging on this
            logger.debug("privacy_config_provider raised %s; treating as ungated", exc)
            return None

    def _privacy_gate_args_json(self, args: Any) -> str:
        """Serialize tool-call args for storage, honouring privacy mode (F076).

        - EPHEMERAL / ISOLATED: elide the payload entirely (no user content
          persists); the row still records counts/latency/status.
        - ANONYMOUS (``pii_redacted``): anonymize string content before the
          usual secret-key redaction + size cap.
        - NORMAL / PUBLIC / no provider: legacy redact-and-cap behaviour.
        """
        config = self._current_privacy_config()
        if config is not None:
            try:
                if config.is_ephemeral() or config.uses_temp_storage():
                    return json_dumps({_CONTENT_GATED_MARKER: config.storage})
                if config.requires_anonymization():
                    return redact_tool_args_json(
                        _anonymize_deep(args, _get_anonymize_text())
                    )
            except Exception as exc:  # noqa: BLE001 - fail closed to gated
                logger.debug(
                    "privacy gate for tool args failed (%s); eliding payload", exc
                )
                return json_dumps({_CONTENT_GATED_MARKER: "error"})
        return redact_tool_args_json(args)

    def _privacy_gate_metadata(
        self, metadata: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Gate a content-bearing metadata dict, honouring privacy mode (F076).

        Same policy as :meth:`_privacy_gate_args_json` but returns a dict for
        the ``metadata`` JSON columns. Operational keys the caller injects for
        forensics (e.g. ``metric_name``) are added by the caller *after* this
        gate, so they are never elided.
        """
        meta = metadata or {}
        config = self._current_privacy_config()
        if config is None:
            return meta
        try:
            if config.is_ephemeral() or config.uses_temp_storage():
                return {_CONTENT_GATED_MARKER: config.storage}
            if config.requires_anonymization():
                return _anonymize_deep(meta, _get_anonymize_text())
        except Exception as exc:  # noqa: BLE001 - fail closed to gated
            logger.debug(
                "privacy gate for metadata failed (%s); eliding payload", exc
            )
            return {_CONTENT_GATED_MARKER: "error"}
        return meta

    def _privacy_gate_text(self, text: Optional[str]) -> Optional[str]:
        """Gate a free-form text field (e.g. ``error_message``) by privacy mode.

        Error/response text can echo user or tool input verbatim, so it is
        content-bearing just like args/metadata (F076):

        - EPHEMERAL / ISOLATED: elide to the gated marker (the row still
          records status/latency so metering and the honesty-layer's read of
          ``result_status`` are unaffected).
        - ANONYMOUS (``pii_redacted``): anonymize the text in place.
        - NORMAL / PUBLIC / no provider: pass through unchanged.
        """
        if text is None:
            return None
        config = self._current_privacy_config()
        if config is None:
            return text
        try:
            if config.is_ephemeral() or config.uses_temp_storage():
                return _CONTENT_GATED_MARKER
            if config.requires_anonymization():
                return _anonymize_deep(text, _get_anonymize_text())
        except Exception as exc:  # noqa: BLE001 - fail closed to gated
            logger.debug("privacy gate for text failed (%s); eliding payload", exc)
            return _CONTENT_GATED_MARKER
        return text

    async def initialize(self) -> None:
        """Create tables if not exists."""
        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        bool_type = self.boolean_type()
        int_pk_type = self.integer_primary_key_type()

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
                agent_did TEXT,
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
        try:
            await self.add_column_if_missing("a2a_llm_calls", "agent_did", "TEXT")
        except Exception as exc:
            logger.debug("Migration check for a2a_llm_calls.agent_did: %s", exc)
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_agent ON a2a_llm_calls(agent_did)"
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

        await self._backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_tool_dispatches (
                id {int_pk_type},
                agent_did TEXT NOT NULL,
                session_id TEXT,
                turn_id TEXT NOT NULL,
                ts {ts_type} {ts_default},
                tool_name TEXT NOT NULL,
                adapter TEXT NOT NULL,
                args_redacted {json_type} NOT NULL,
                result_status TEXT NOT NULL CHECK (
                    result_status IN ('success', 'error', 'empty', 'policy_denied', 'timeout')
                ),
                error_class TEXT,
                error_message TEXT,
                latency_ms INTEGER NOT NULL,
                result_size_bytes INTEGER
            )
        """)
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_dispatches_agent_turn "
            "ON a2a_tool_dispatches(agent_did, turn_id, id)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_dispatches_agent_ts "
            "ON a2a_tool_dispatches(agent_did, ts)"
        )
        await self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_dispatches_status "
            "ON a2a_tool_dispatches(agent_did, result_status)"
        )

        logger.info(f"ObservabilityStore initialized ({self._backend.backend_type})")

    async def log_tool_dispatch(self, entry: ToolDispatchEntry) -> None:
        """Best-effort structured insert. Never break dispatch on log failure."""
        try:
            await self._backend.execute(
                """
                INSERT INTO a2a_tool_dispatches (
                    agent_did, session_id, turn_id, ts, tool_name, adapter,
                    args_redacted, result_status, error_class, error_message,
                    latency_ms, result_size_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.agent_did,
                    entry.session_id,
                    entry.turn_id,
                    self.now_utc_param(),
                    entry.tool_name,
                    entry.adapter,
                    self._privacy_gate_args_json(entry.args_redacted),
                    entry.result_status,
                    entry.error_class,
                    _cap_text(
                        self._privacy_gate_text(entry.error_message),
                        MAX_TOOL_ERROR_MESSAGE_CHARS,
                    ),
                    int(entry.latency_ms),
                    entry.result_size_bytes,
                ),
            )
        except Exception as exc:
            print(f"a2a_tool_dispatches write failed: {exc}", file=sys.stderr)

    async def log_structured_tool_call(self, entry: ToolDispatchEntry) -> None:
        """Compatibility alias for callers/tests using the original #1239 name."""
        await self.log_tool_dispatch(entry)

    async def tool_failure_rate(
        self, agent_did: str, last_n_turns: int = 100
    ) -> dict[str, Any]:
        """Return per-tool/per-error counts and rates for recent turns."""
        turn_rows = await self._backend.fetch_all(
            """
            SELECT turn_id
            FROM (
                SELECT turn_id, MAX(id) AS last_id
                FROM a2a_tool_dispatches
                WHERE agent_did = ? AND turn_id IS NOT NULL
                GROUP BY turn_id
                ORDER BY last_id DESC
                LIMIT ?
            )
            """,
            (agent_did, int(last_n_turns)),
        )
        turn_ids = [row[0] for row in turn_rows]
        if not turn_ids:
            return {
                "agent_did": agent_did,
                "last_n_turns": int(last_n_turns),
                "turns_observed": 0,
                "total_calls": 0,
                "failure_calls": 0,
                "failure_rate": 0.0,
                "dominant_failures": [],
            }

        placeholders = ", ".join("?" for _ in turn_ids)
        params = (agent_did, *turn_ids)
        total_row = await self._backend.fetch_one(
            f"""
            SELECT COUNT(*)
            FROM a2a_tool_dispatches
            WHERE agent_did = ? AND turn_id IN ({placeholders})
            """,
            params,
        )
        failures = await self._backend.fetch_all(
            f"""
            SELECT
                tool_name,
                COALESCE(error_class, result_status) AS error_class,
                COUNT(*) AS count
            FROM a2a_tool_dispatches
            WHERE agent_did = ?
              AND turn_id IN ({placeholders})
              AND result_status != 'success'
            GROUP BY tool_name, COALESCE(error_class, result_status)
            ORDER BY count DESC, tool_name ASC, error_class ASC
            """,
            params,
        )
        failure_total = sum(int(row[2]) for row in failures)
        total_calls = int(total_row[0] if total_row else 0)
        return {
            "agent_did": agent_did,
            "last_n_turns": int(last_n_turns),
            "turns_observed": len(turn_ids),
            "total_calls": total_calls,
            "failure_calls": failure_total,
            "failure_rate": failure_total / total_calls if total_calls else 0.0,
            "dominant_failures": [
                {
                    "tool_name": row[0],
                    "error_class": row[1],
                    "count": int(row[2]),
                    "rate": int(row[2]) / total_calls if total_calls else 0.0,
                }
                for row in failures
            ],
        }

    async def recent_failures(
        self, agent_did: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self._backend.fetch_all(
            """
            SELECT
                id, agent_did, session_id, turn_id, ts, tool_name, adapter,
                args_redacted, result_status, error_class, error_message,
                latency_ms, result_size_bytes
            FROM a2a_tool_dispatches
            WHERE agent_did = ? AND result_status != 'success'
            ORDER BY id DESC
            LIMIT ?
            """,
            (agent_did, int(limit)),
        )
        return [
            {
                "id": row[0],
                "agent_did": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "ts": str(row[4]) if row[4] is not None else None,
                "tool_name": row[5],
                "adapter": row[6],
                "args_redacted": json_loads(row[7]) if row[7] else {},
                "result_status": row[8],
                "error_class": row[9],
                "error_message": row[10],
                "latency_ms": row[11],
                "result_size_bytes": row[12],
            }
            for row in rows
        ]

    async def purge_observability_since(
        self,
        since_iso: str,
        *,
        agent_did: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> dict[str, int]:
        """Safety-net sweep for the EPHEMERAL leak-purge (F076).

        Deletes ``a2a_tool_dispatches`` and ``a2a_observability`` rows authored
        on/after ``since_iso`` — the watermark recorded when the agent entered
        EPHEMERAL. This is the inverse of the privacy gate above: the gate stops
        content-bearing payloads from landing during the stint; this sweep
        scrubs any that slipped through (or predated the gate being wired up).

        The watermark from ``PrivacyEnforcingStorage`` uses SQLite's
        ``datetime('now')`` shape (space separator, no offset), but the store
        writes ISO-8601 timestamps (``T`` separator, offset). A raw
        lexicographic ``>=`` between the two would mis-order at the separator
        (space ``0x20`` vs ``T`` ``0x54``) and over-delete — the same class of
        bug #867 documents. So we normalize the watermark to the store's own
        timestamp form before comparing.

        Scoped by ``agent_did`` (tool dispatches) / ``agent_name``
        (observability) when provided so a shared PostgreSQL backend never
        reaches across tenants. Returns ``{table: rows_deleted}``.
        """
        since_param = self._normalize_since_param(since_iso)
        result: dict[str, int] = {}

        dispatch_where = "ts >= ?"
        dispatch_params: list[Any] = [since_param]
        if agent_did:
            dispatch_where += " AND agent_did = ?"
            dispatch_params.append(agent_did)
        try:
            result["a2a_tool_dispatches"] = await self._backend.execute(
                f"DELETE FROM a2a_tool_dispatches WHERE {dispatch_where}",
                tuple(dispatch_params),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort safety net
            logger.warning("purge_observability_since: tool-dispatch sweep failed: %s", exc)
            result["a2a_tool_dispatches"] = 0

        obs_where = "timestamp >= ?"
        obs_params: list[Any] = [since_param]
        if agent_name:
            obs_where += " AND agent_name = ?"
            obs_params.append(agent_name)
        try:
            result["a2a_observability"] = await self._backend.execute(
                f"DELETE FROM a2a_observability WHERE {obs_where}",
                tuple(obs_params),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort safety net
            logger.warning("purge_observability_since: observability sweep failed: %s", exc)
            result["a2a_observability"] = 0

        return result

    def _normalize_since_param(self, since_iso: str) -> Any:
        """Coerce a watermark string into the store's timestamp parameter form.

        Accepts both the ``YYYY-MM-DD HH:MM:SS`` watermark shape and a full
        ISO-8601 string; returns a value comparable against the rows this store
        writes (``datetime`` for PostgreSQL, ISO string for SQLite).
        """
        raw = str(since_iso).strip()
        try:
            dt = datetime.fromisoformat(raw.replace(" ", "T"))
        except ValueError:
            return since_iso
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return self.to_timestamp_param(dt)

    async def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Log start of tool call. Returns event_id for timing.

        ``timestamp`` lets an external producer preserve the original event
        time (e.g. telemetry pushed after the fact); defaults to now.
        """
        event_id = generate_id()
        now = self.to_timestamp_param(timestamp) if timestamp else self.now_utc_param()

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
                json_dumps(self._privacy_gate_metadata(metadata)),
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
            (
                self.to_bool_param(success),
                duration_ms,
                self._privacy_gate_text(error_message),
                event_id,
            ),
        )

    async def log_agent_response(
        self,
        agent_name: str,
        duration_ms: int,
        success: bool = True,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Log an agent response. Returns event_id.

        ``timestamp`` preserves an external producer's original event time;
        defaults to now.
        """
        event_id = generate_id()
        now = self.to_timestamp_param(timestamp) if timestamp else self.now_utc_param()

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
                json_dumps(self._privacy_gate_metadata(metadata)),
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
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Log an error event. Returns event_id.

        ``timestamp`` preserves an external producer's original event time;
        defaults to now.
        """
        event_id = generate_id()
        # Gate the caller's content-bearing metadata, THEN stamp the
        # operational key so it always survives the gate (F076).
        meta = self._privacy_gate_metadata(metadata)
        meta["error_type"] = error_type
        now = self.to_timestamp_param(timestamp) if timestamp else self.now_utc_param()

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
                self._privacy_gate_text(error_message),
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
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Log a metric value. Returns event_id.

        ``timestamp`` preserves an external producer's original event time;
        defaults to now.
        """
        event_id = generate_id()
        # Gate the caller's content-bearing metadata, THEN stamp the
        # operational keys so name/value always survive the gate (F076).
        meta = self._privacy_gate_metadata(metadata)
        meta["metric_name"] = metric_name
        meta["metric_value"] = metric_value
        now = self.to_timestamp_param(timestamp) if timestamp else self.now_utc_param()

        await self._backend.execute(
            """
            INSERT INTO a2a_observability
            (id, timestamp, agent_name, session_id, event_type, metadata)
            VALUES (?, ?, ?, ?, 'metric', ?)
            """,
            (event_id, now, agent_name, session_id, json_dumps(meta)),
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
        metadata_equals: Optional[tuple[str, str]] = None,
    ) -> list[ObservabilityEvent]:
        """Query observability events with filters.

        ``metadata_equals`` is a ``(json_key, value)`` pair that matches a
        top-level string field inside the JSON ``metadata`` column (e.g.
        ``("metric_name", "assistant_turn_persist_failed")``). It pushes the
        filter into SQL so ``limit`` applies to the matching rows rather than
        to every row of the ``event_type``. The predicate uses each backend's
        native JSON accessor — ``metadata ->> ?`` on PostgreSQL (JSONB),
        ``json_extract(metadata, ?)`` on SQLite — so it is an exact, portable
        structural match (not a substring ``LIKE``) on both backends.
        """
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
        if metadata_equals:
            json_key, json_value = metadata_equals
            if self.is_postgres:
                conditions.append("metadata ->> ? = ?")
                params.append(json_key)
            else:
                conditions.append("json_extract(metadata, ?) = ?")
                params.append(f"$.{json_key}")
            params.append(json_value)
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

    async def get_metric_summary(
        self,
        metric_name: str,
        agent_name: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 5000,
        sample_limit: int = 20,
    ) -> dict[str, Any]:
        """Summarize a single named metric over a window (#969).

        Metrics are stored as ``event_type='metric'`` rows with the metric name
        inside the JSON ``metadata`` (there is no dedicated column). Consistent
        with how this store reads all JSON metadata in Python, we fetch the
        metric events in the window and aggregate by name in Python — portable
        across the SQLite and Postgres backends with no backend-specific JSON
        SQL. Surfaces otherwise-"dark" forensic metrics such as
        ``assistant_turn_persist_failed``.

        Returns ``metric_name``, ``count``, ``total_value``, ``first_seen``,
        ``last_seen``, ``by_agent`` counts, recent ``samples`` (each carrying
        the metric's forensic metadata, e.g. ``error_type``/``session_id``),
        and ``truncated`` (True if the window held more than ``limit`` metric
        events, so ``count`` is a lower bound).
        """
        # Push the metric-name filter into SQL so ``limit`` bounds THIS metric's
        # rows, not all metric events. Without this, a rare forensic metric
        # (e.g. assistant_turn_persist_failed) would be missed entirely when a
        # high-volume metric (feature_tools_built_streaming) fills the newest
        # ``limit`` rows. The parsed-metadata check below stays authoritative (#969).
        events = await self.query_events(
            agent_name=agent_name,
            event_type="metric",
            since=since,
            until=until,
            limit=limit + 1,
            metadata_equals=("metric_name", metric_name),
        )
        truncated = len(events) > limit
        if truncated:
            events = events[:limit]

        matched = [
            e for e in events if (e.metadata or {}).get("metric_name") == metric_name
        ]

        total_value = 0.0
        for e in matched:
            try:
                total_value += float((e.metadata or {}).get("metric_value") or 0)
            except (TypeError, ValueError):
                pass

        by_agent: dict[str, int] = {}
        for e in matched:
            by_agent[e.agent_name] = by_agent.get(e.agent_name, 0) + 1

        timestamps = [e.timestamp for e in matched if e.timestamp is not None]
        # query_events returns newest-first, so ``matched`` is already DESC.
        samples = [
            {
                "timestamp": str(e.timestamp),
                "agent_name": e.agent_name,
                "value": (e.metadata or {}).get("metric_value"),
                "metadata": {
                    k: v
                    for k, v in (e.metadata or {}).items()
                    if k not in ("metric_name", "metric_value")
                },
            }
            for e in matched[:sample_limit]
        ]

        return {
            "metric_name": metric_name,
            "count": len(matched),
            "total_value": total_value,
            "first_seen": str(min(timestamps)) if timestamps else None,
            "last_seen": str(max(timestamps)) if timestamps else None,
            "by_agent": by_agent,
            "samples": samples,
            "truncated": truncated,
        }

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
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(older_than_days))
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
        rows_affected = await self._backend.execute(
            "DELETE FROM a2a_observability WHERE timestamp < ?",
            (cutoff_str,),
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
        agent_did: Optional[str] = None,
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
            agent_did: Agent DID that owns the LLM call

        Returns:
            event_id for the logged call
        """
        event_id = generate_id()
        now = self.now_utc_param()

        # Privacy-gate content-bearing fields FIRST (F076): prompts/response
        # previews, error text, tool-call args, and metadata all echo user
        # content, so they must honour the agent's privacy mode exactly like
        # the tool-call sink does. EPHEMERAL/ISOLATED elide to the gated
        # marker, ANONYMOUS anonymizes; counts/latency/model always persist.
        system_prompt = self._privacy_gate_text(system_prompt)
        user_prompt = self._privacy_gate_text(user_prompt)
        response = self._privacy_gate_text(response)
        error_message = self._privacy_gate_text(error_message)
        metadata = self._privacy_gate_metadata(metadata)
        tool_calls_json = (
            self._privacy_gate_args_json(tool_calls) if tool_calls else None
        )

        # Truncate prompts/response to preview (first 500 chars)
        preview_len = 500
        system_preview = system_prompt[:preview_len] if system_prompt else None
        user_preview = user_prompt[:preview_len] if user_prompt else None
        response_preview = response[:preview_len] if response else None

        await self._backend.execute(
            """
            INSERT INTO a2a_llm_calls
            (id, timestamp, agent_did, session_id, companion_id, user_id, provider, model,
             system_prompt_preview, user_prompt_preview, response_preview,
             input_tokens, output_tokens, duration_ms, success, error_message,
             tool_calls, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                now,
                agent_did,
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
                tool_calls_json,
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
        agent_did: Optional[str] = None,
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
            agent_did: Filter by agent DID
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
        if agent_did:
            conditions.append("agent_did = ?")
            params.append(agent_did)
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
            SELECT
                id, timestamp, agent_did, session_id, companion_id, user_id,
                provider, model, system_prompt_preview, user_prompt_preview,
                response_preview, input_tokens, output_tokens, duration_ms,
                success, error_message, tool_calls, metadata
            FROM a2a_llm_calls
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
        0: id, 1: timestamp, 2: agent_did, 3: session_id, 4: companion_id,
        5: user_id, 6: provider, 7: model, 8: system_prompt_preview,
        9: user_prompt_preview, 10: response_preview, 11: input_tokens,
        12: output_tokens, 13: duration_ms, 14: success, 15: error_message,
        16: tool_calls, 17: metadata
        """
        return LLMCallEvent(
            event_id=row[0],
            timestamp=self.from_timestamp_field(row[1]),
            agent_did=row[2],
            session_id=row[3],
            companion_id=row[4],
            user_id=row[5],
            provider=row[6],
            model=row[7],
            system_prompt_preview=row[8],
            user_prompt_preview=row[9],
            response_preview=row[10],
            input_tokens=row[11],
            output_tokens=row[12],
            duration_ms=row[13],
            success=self.from_bool_field(row[14]),
            error_message=row[15],
            tool_calls=json_loads(row[16]) if row[16] else None,
            metadata=json_loads(row[17]) if row[17] else {},
        )


def redact_tool_args_json(value: Any) -> str:
    """Redact secret-looking fields and cap serialized JSON at ~2 KiB."""
    redacted = _redact(value)
    text = json_dumps(redacted)
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_TOOL_ARGS_JSON_BYTES:
        return text
    budget = MAX_TOOL_ARGS_JSON_BYTES - 128
    preview = encoded[: max(0, budget)].decode("utf-8", errors="ignore")
    return json_dumps({"_truncated": True, "preview": preview})


def _anonymize_deep(value: Any, anonymize: Callable[[str], str], depth: int = 0) -> Any:
    """Apply PII anonymization to every string leaf of a nested structure (F076).

    Structure (dict/list nesting) is preserved so downstream redaction and size
    capping still see a normal payload; only the human-readable string content
    is scrubbed. Secret-key redaction runs afterwards in ``redact_tool_args_json``.
    """
    if depth > 8:
        return value
    if isinstance(value, dict):
        return {str(k): _anonymize_deep(v, anonymize, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_anonymize_deep(item, anonymize, depth + 1) for item in value]
    if isinstance(value, str):
        try:
            return anonymize(value)
        except Exception:  # noqa: BLE001 - never break logging on anonymizer error
            return value
    return value


def infer_tool_result_status(result: Any, status_hint: Optional[str] = None) -> str:
    if status_hint:
        return status_hint
    if result is None or result == "" or result == [] or result == {}:
        return "empty"
    if isinstance(result, dict):
        success = result.get("success")
        status = str(result.get("status", "")).lower()
        error_text = str(result.get("error", ""))
        if error_text.lower().startswith("permission denied"):
            return "policy_denied"
        if success is False or status in {"error", "failed", "failure"} or result.get("error"):
            return "error"
    if getattr(result, "failed", False):
        return "error"
    error = getattr(result, "error", None)
    if error:
        return "error"
    status = str(getattr(result, "status", "")).lower()
    if status in {"error", "failed", "failure"}:
        return "error"
    return "success"


def tool_result_size_bytes(result: Any) -> Optional[int]:
    try:
        from kestrel_sovereign.features.base import _serialize_tool_result

        return len(json_dumps(_serialize_tool_result(result)).encode("utf-8"))
    except Exception:
        try:
            return len(str(result).encode("utf-8"))
        except Exception:
            return None


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            key_str = str(key)
            if _is_secret_key(key_str):
                out[key_str] = "<redacted>"
            else:
                out[key_str] = _redact(val, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 50:
            return [_redact(item, depth + 1) for item in value[:50]] + ["<truncated>"]
        return [_redact(item, depth + 1) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + "...<truncated>"
    return value


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(secret in normalized for secret in _SECRET_KEYS)


def _cap_text(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit]
