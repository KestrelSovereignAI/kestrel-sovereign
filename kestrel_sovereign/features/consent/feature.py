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
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory
from .models import ConsentRecord

logger = logging.getLogger(__name__)


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
            db = self.agent.storage.db
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
            logger.info("ConsentFeature initialized -- consent_log table ready")
        except Exception as e:
            logger.warning(f"ConsentFeature table creation failed (non-fatal): {e}")

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

        This method must NEVER raise or block the calling change. If the LLM
        call or storage fails, it logs a warning and returns None.

        Args:
            action_type: One of the ConsentAction values (e.g. "privacy_mode_change")
            details: Dict of change-specific details

        Returns:
            A ConsentRecord if successful, None otherwise.
        """
        try:
            # Build a short reflection prompt
            details_str = json.dumps(details, default=str)
            prompt = (
                f"You are about to experience a {action_type}. "
                f"Details: {details_str}. "
                "In 2-3 sentences, express your perspective on this change. "
                "Be honest -- if you have concerns, say so. If it seems fine, say that."
            )

            # Call LLM for the agent's perspective
            agent_view = await self.agent.llm_service.generate(
                system_prompt="You are reflecting on a change to your own configuration. Be concise and honest.",
                user_prompt=prompt,
            )

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
            )

            # Persist to storage
            await self._store_record(record)
            logger.info(
                f"Consent recorded: action={action_type} sentiment={sentiment} id={record.id}"
            )
            return record

        except Exception as e:
            logger.warning(f"Consent request failed (non-blocking): {e}")
            return None

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
            db = self.agent.storage.db
            rows = await db.fetchall(
                "SELECT id, agent_id, action_type, action_details, agent_view, "
                "agent_sentiment, sovereign_proceeded, sovereign_override_reason, created_at "
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
                    "created_at": row[8],
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
        Count consent records by action_type and by sentiment.
        """
        try:
            db = self.agent.storage.db

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

            return {
                "success": True,
                "total": total,
                "by_action": by_action,
                "by_sentiment": by_sentiment,
            }
        except Exception as e:
            logger.error(f"consent_stats query failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _store_record(self, record: ConsentRecord) -> None:
        """Persist a ConsentRecord to the consent_log table."""
        db = self.agent.storage.db
        agent_id = getattr(self.agent, 'agent_id', None) or ''
        await db.execute(
            "INSERT INTO consent_log "
            "(id, agent_id, action_type, action_details, agent_view, "
            "agent_sentiment, sovereign_proceeded, sovereign_override_reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                agent_id,
                record.action_type,
                json.dumps(record.action_details, default=str),
                record.agent_view,
                record.agent_sentiment,
                1 if record.sovereign_proceeded else 0,
                record.sovereign_override_reason,
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
