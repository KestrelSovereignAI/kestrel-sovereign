"""
Agent Consent Protocol Feature.

Gives the agent a recorded voice before significant changes take effect
(privacy mode switch, model change, safe mode entry). The Sovereign retains
full authority -- this is a voice, not a veto.

Each consent request:
1. Asks the agent to briefly reflect on the proposed change
2. Parses a sentiment from the response
3. Stores the record in a consent_log table
4. Returns the record (or None on failure -- never blocks the change)

The LLM call is timeboxed with a hard timeout (default 5s). On timeout or
error, the change proceeds (fail-open). Duration and timeout metrics are
tracked in the consent_log table and exposed via !consent-stats.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.tools.base import ToolCategory
from .models import ConsentRecord

logger = logging.getLogger(__name__)

# Hard timeout for consent LLM calls. The change always proceeds regardless.
CONSENT_TIMEOUT_SECONDS = 5.0


class ConsentFeature(Feature):
    """
    Feature that records the agent's perspective before significant changes.

    Provides:
    - request_consent(): called by other features before changes
    - !consent-log: view recent consent records
    - !consent-stats: view consent statistics by action and sentiment
    """

    @property
    def tool_description(self) -> str:
        return (
            "Agent consent protocol - records the agent's perspective before "
            "significant changes like privacy mode switches, model changes, "
            "and safe mode entry. View consent history and statistics."
        )

    async def initialize(self):
        """Create the consent_log table if it does not exist."""
        try:
            db = resolve_feature_database(self.agent)
            if db is None:
                raise RuntimeError("database not available")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS consent_log (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    action_type TEXT NOT NULL,
                    action_details TEXT NOT NULL,
                    agent_view TEXT NOT NULL,
                    agent_sentiment TEXT DEFAULT 'neutral',
                    sovereign_proceeded INTEGER DEFAULT 1,
                    sovereign_override_reason TEXT,
                    duration_ms REAL,
                    timed_out INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_consent_log_action
                ON consent_log(action_type)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_consent_log_created
                ON consent_log(created_at DESC)
            """)
            # Migrate existing tables that lack the new columns
            await self._migrate_add_metrics_columns(db)
            logger.info("ConsentFeature initialized -- consent_log table ready")
        except Exception as e:
            logger.warning(f"ConsentFeature table creation failed (non-fatal): {e}")

    async def _migrate_add_metrics_columns(self, db) -> None:
        """Add duration_ms and timed_out columns if they don't exist yet."""
        for col, col_def in [("duration_ms", "REAL"), ("timed_out", "INTEGER DEFAULT 0")]:
            try:
                await db.execute(
                    f"ALTER TABLE consent_log ADD COLUMN {col} {col_def}"
                )
            except Exception:
                # Column already exists -- expected on non-first run
                pass

    # =========================================================================
    # Core consent API (called by other features)
    # =========================================================================

    async def request_consent(
        self,
        action_type: str,
        details: dict,
    ) -> Optional[ConsentRecord]:
        """
        Ask the agent to reflect on a proposed change and record the response.

        This method must NEVER raise or block the calling change. The LLM call
        is wrapped in a hard timeout (CONSENT_TIMEOUT_SECONDS). On timeout or
        any error the method returns None and the change proceeds (fail-open).

        Args:
            action_type: One of the ConsentAction values (e.g. "privacy_mode_change")
            details: Dict of change-specific details

        Returns:
            A ConsentRecord if successful, None otherwise.
        """
        try:
            record = await asyncio.wait_for(
                self._generate_consent(action_type, details),
                timeout=CONSENT_TIMEOUT_SECONDS,
            )
            return record
        except asyncio.TimeoutError:
            logger.warning(
                f"Consent request timed out after {CONSENT_TIMEOUT_SECONDS}s "
                f"for {action_type} -- proceeding (fail-open)"
            )
            await self._record_timeout(action_type, details)
            return None
        except Exception as e:
            logger.warning(f"Consent request failed for {action_type}: {e} -- proceeding (fail-open)")
            return None

    async def _generate_consent(
        self,
        action_type: str,
        details: dict,
    ) -> ConsentRecord:
        """
        Internal method that performs the LLM call and builds the ConsentRecord.

        Tracks wall-clock duration of the LLM call and stores it in the record.
        This method may raise; the caller (request_consent) handles errors.
        """
        # Build a short reflection prompt
        details_str = json.dumps(details, default=str)
        prompt = (
            f"You are about to experience a {action_type}. "
            f"Details: {details_str}. "
            "In 2-3 sentences, express your perspective on this change. "
            "Be honest -- if you have concerns, say so. If it seems fine, say that."
        )

        # Call LLM for the agent's perspective, tracking duration
        start = time.monotonic()
        agent_view = await self.agent.llm_service.generate(
            system_prompt="You are reflecting on a change to your own configuration. Be concise and honest.",
            user_prompt=prompt,
        )
        duration_ms = (time.monotonic() - start) * 1000

        # Handle LLMResponse objects (may have .content attribute)
        if hasattr(agent_view, 'content'):
            agent_view = agent_view.content or ""
        if not isinstance(agent_view, str):
            agent_view = str(agent_view)

        # Parse sentiment from the response
        sentiment = self._parse_sentiment(agent_view)

        # Build the record
        record = ConsentRecord(
            id=uuid.uuid4().hex[:12],
            action_type=action_type,
            action_details=details,
            agent_view=agent_view,
            agent_sentiment=sentiment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            timed_out=False,
        )

        # Persist to storage
        await self._store_record(record)
        logger.info(
            f"Consent recorded: action={action_type} sentiment={sentiment} "
            f"duration={duration_ms:.0f}ms id={record.id}"
        )
        return record

    async def _record_timeout(self, action_type: str, details: dict) -> None:
        """
        Persist a timeout record to consent_log so it shows up in stats.

        Failures here are silently swallowed -- we never block the change.
        """
        try:
            record = ConsentRecord(
                id=uuid.uuid4().hex[:12],
                action_type=action_type,
                action_details=details,
                agent_view="[TIMEOUT]",
                agent_sentiment="timeout",
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=CONSENT_TIMEOUT_SECONDS * 1000,
                timed_out=True,
            )
            await self._store_record(record)
            logger.info(f"Consent timeout recorded: action={action_type} id={record.id}")
        except Exception as e:
            logger.warning(f"Failed to record consent timeout (non-blocking): {e}")

    # =========================================================================
    # Tool commands
    # =========================================================================

    @tool(
        name="consent_log",
        description="View recent consent records showing the agent's perspective on past changes.",
        category=ToolCategory.SYSTEM,
        command_prefix="!consent-log",
    )
    async def consent_log(self, limit: int = 10) -> Dict[str, Any]:
        """
        Query the last N consent records.

        Args:
            limit: Maximum number of records to return (default 10)
        """
        try:
            db = resolve_feature_database(self.agent)
            if db is None:
                raise RuntimeError("database not available")
            rows = await db.fetchall(
                "SELECT id, agent_id, action_type, action_details, agent_view, "
                "agent_sentiment, sovereign_proceeded, sovereign_override_reason, "
                "duration_ms, timed_out, created_at "
                "FROM consent_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            records = []
            for row in rows:
                records.append({
                    "id": row[0],
                    "agent_id": row[1],
                    "action_type": row[2],
                    "action_details": row[3],
                    "agent_view": row[4],
                    "agent_sentiment": row[5],
                    "sovereign_proceeded": bool(row[6]),
                    "sovereign_override_reason": row[7],
                    "duration_ms": row[8],
                    "timed_out": bool(row[9]) if row[9] is not None else False,
                    "created_at": row[10],
                })
            return {
                "success": True,
                "records": records,
                "count": len(records),
            }
        except Exception as e:
            logger.error(f"consent_log query failed: {e}")
            return {"success": False, "error": str(e)}

    @tool(
        name="consent_stats",
        description="View consent statistics grouped by action type and sentiment.",
        category=ToolCategory.SYSTEM,
        command_prefix="!consent-stats",
    )
    async def consent_stats(self) -> Dict[str, Any]:
        """
        Count consent records by action_type and by sentiment, plus
        latency and timeout/error metrics.
        """
        try:
            db = resolve_feature_database(self.agent)
            if db is None:
                raise RuntimeError("database not available")

            # Counts by action type
            action_rows = await db.fetchall(
                "SELECT action_type, COUNT(*) FROM consent_log GROUP BY action_type"
            )
            by_action = {row[0]: row[1] for row in action_rows}

            # Counts by sentiment
            sentiment_rows = await db.fetchall(
                "SELECT agent_sentiment, COUNT(*) FROM consent_log GROUP BY agent_sentiment"
            )
            by_sentiment = {row[0]: row[1] for row in sentiment_rows}

            # Total
            total_row = await db.fetchone("SELECT COUNT(*) FROM consent_log")
            total = total_row[0] if total_row else 0

            # ---- Timing and reliability metrics ----

            # Average duration (exclude NULL for old records without tracking)
            avg_row = await db.fetchone(
                "SELECT AVG(duration_ms) FROM consent_log WHERE duration_ms IS NOT NULL"
            )
            avg_duration_ms = round(avg_row[0], 1) if avg_row and avg_row[0] is not None else None

            # P95 duration (approximate: order by duration, pick 95th percentile row)
            p95_duration_ms = None
            duration_count_row = await db.fetchone(
                "SELECT COUNT(*) FROM consent_log WHERE duration_ms IS NOT NULL"
            )
            duration_count = duration_count_row[0] if duration_count_row else 0
            if duration_count > 0:
                p95_offset = max(0, int(duration_count * 0.95) - 1)
                p95_row = await db.fetchone(
                    "SELECT duration_ms FROM consent_log "
                    "WHERE duration_ms IS NOT NULL "
                    "ORDER BY duration_ms ASC LIMIT 1 OFFSET ?",
                    (p95_offset,),
                )
                p95_duration_ms = round(p95_row[0], 1) if p95_row and p95_row[0] is not None else None

            # Timeout count and rate
            timeout_row = await db.fetchone(
                "SELECT COUNT(*) FROM consent_log WHERE timed_out = 1"
            )
            timeout_count = timeout_row[0] if timeout_row else 0
            timeout_rate = round(timeout_count / total, 4) if total > 0 else 0.0

            # Error count (sentiment = 'timeout' covers timeouts; agent_view
            # starting with '[' covers system-generated entries)
            error_row = await db.fetchone(
                "SELECT COUNT(*) FROM consent_log WHERE timed_out = 1"
            )
            error_count = error_row[0] if error_row else 0
            error_rate = round(error_count / total, 4) if total > 0 else 0.0

            return {
                "success": True,
                "total": total,
                "by_action": by_action,
                "by_sentiment": by_sentiment,
                "avg_duration_ms": avg_duration_ms,
                "p95_duration_ms": p95_duration_ms,
                "timeout_count": timeout_count,
                "timeout_rate": timeout_rate,
                "error_count": error_count,
                "error_rate": error_rate,
            }
        except Exception as e:
            logger.error(f"consent_stats query failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _store_record(self, record: ConsentRecord) -> None:
        """Persist a ConsentRecord to the consent_log table."""
        db = resolve_feature_database(self.agent)
        if db is None:
            raise RuntimeError("database not available")
        agent_id = self.agent.did
        await db.execute(
            "INSERT INTO consent_log "
            "(id, agent_id, action_type, action_details, agent_view, "
            "agent_sentiment, sovereign_proceeded, sovereign_override_reason, "
            "duration_ms, timed_out, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                agent_id,
                record.action_type,
                json.dumps(record.action_details, default=str),
                record.agent_view,
                record.agent_sentiment,
                1 if record.sovereign_proceeded else 0,
                record.sovereign_override_reason,
                record.duration_ms,
                1 if record.timed_out else 0,
                record.timestamp,
            ),
        )

    @staticmethod
    def _parse_sentiment(text: str) -> str:
        """
        Derive a simple sentiment label from the agent's response text.

        Returns one of: positive, negative, neutral, concerned.
        """
        lower = text.lower()

        concern_words = [
            "concern", "worried", "uneasy", "cautious", "risk",
            "careful", "hesitant", "uncertain", "wary",
        ]
        negative_words = [
            "disagree", "oppose", "object", "against", "bad",
            "harmful", "dangerous", "wrong", "dislike",
        ]
        positive_words = [
            "agree", "good", "fine", "happy", "welcome", "great",
            "positive", "support", "approve", "makes sense",
            "reasonable", "understood",
        ]

        concern_score = sum(1 for w in concern_words if w in lower)
        negative_score = sum(1 for w in negative_words if w in lower)
        positive_score = sum(1 for w in positive_words if w in lower)

        if negative_score > positive_score and negative_score > concern_score:
            return "negative"
        if concern_score > positive_score:
            return "concerned"
        if positive_score > 0:
            return "positive"
        return "neutral"
