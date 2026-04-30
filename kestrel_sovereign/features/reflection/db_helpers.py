"""
Database helper functions for reflection feature.

Handles storage and retrieval of:
- Insights
- Reflection sessions
- Improvement proposals
- Behavior rules
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .models import (
    Insight,
    InsightType,
    ReflectionSession,
    ImprovementProposal,
    BehaviorRule,
    ChangeType,
)

logger = logging.getLogger(__name__)


class ReflectionDatabaseHelper:
    """Helper class for reflection feature database operations."""

    def __init__(self, db, agent_id: str):
        """
        Initialize the database helper.

        Args:
            db: Database connection/interface
            agent_id: Agent identifier for scoping queries
        """
        self.db = db
        self.agent_id = agent_id

    async def store_insight(self, insight: Insight, session_id: str) -> None:
        """Store an insight to the database."""
        if not self.db:
            return

        await self.db.execute(
            """
            INSERT INTO reflection_insights
            (id, agent_id, session_id, type, title, description, evidence,
             confidence, actionable, suggested_action, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                insight.id,
                self.agent_id,
                session_id,
                insight.type.value,
                insight.title,
                insight.description,
                json.dumps(insight.evidence),
                insight.confidence,
                1 if insight.actionable else 0,
                insight.suggested_action,
                insight.created_at.isoformat(),
            ),
        )

    async def store_session(self, session: ReflectionSession) -> None:
        """Store a reflection session to the database."""
        if not self.db:
            return

        await self.db.execute(
            """
            INSERT INTO reflection_sessions
            (id, agent_id, trigger, started_at, completed_at, interactions_analyzed,
             episodes_analyzed, insights_generated, improvements_proposed,
             improvements_approved, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                self.agent_id,
                session.trigger,
                session.started_at.isoformat() if session.started_at else None,
                session.completed_at.isoformat() if session.completed_at else None,
                session.interactions_analyzed,
                session.episodes_analyzed,
                session.insights_count,
                session.improvements_proposed,
                session.improvements_approved,
                session.error,
            ),
        )

    async def store_proposal(self, proposal: ImprovementProposal) -> None:
        """Store an improvement proposal to the database."""
        if not self.db:
            return

        await self.db.execute(
            """
            INSERT INTO improvement_proposals
            (id, agent_id, insight_id, title, description, change_type,
             proposed_change, requires_approval, approved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.id,
                self.agent_id,
                proposal.insight_id,
                proposal.title,
                proposal.description,
                proposal.change_type.value,
                proposal.proposed_change,
                1 if proposal.requires_approval else 0,
                1 if proposal.approved else 0,
                proposal.created_at.isoformat(),
            ),
        )

    async def update_proposal(self, proposal: ImprovementProposal) -> None:
        """Update an improvement proposal in the database."""
        if not self.db:
            return

        await self.db.execute(
            """
            UPDATE improvement_proposals
            SET approved = ?, rejection_reason = ?, approved_at = ?,
                approved_by = ?, applied_at = ?
            WHERE id = ?
            """,
            (
                1 if proposal.approved else 0,
                proposal.rejection_reason,
                proposal.approved_at.isoformat() if proposal.approved_at else None,
                proposal.approved_by,
                proposal.applied_at.isoformat() if proposal.applied_at else None,
                proposal.id,
            ),
        )

    async def store_behavior_rule(self, rule: BehaviorRule) -> None:
        """Store a behavior rule to the database."""
        if not self.db:
            return

        await self.db.execute(
            """
            INSERT INTO behavior_rules
            (id, agent_id, proposal_id, trigger_condition, action,
             change_type, active, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule.id,
                self.agent_id,
                rule.proposal_id,
                rule.trigger_condition,
                rule.action,
                rule.change_type.value,
                1 if rule.active else 0,
                rule.priority,
                rule.created_at.isoformat(),
            ),
        )

    async def get_insights(
        self,
        type_filter: str = None,
        min_confidence: float = 0.5,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get insights from the database.

        Args:
            type_filter: Filter by insight type
            min_confidence: Minimum confidence threshold
            limit: Maximum number of insights to return

        Returns:
            List of insight dictionaries
        """
        if not self.db:
            return []

        # Build query
        query = """
            SELECT id, type, title, description, evidence, confidence,
                   actionable, suggested_action, created_at
            FROM reflection_insights
            WHERE agent_id = ? AND confidence >= ?
        """
        params = [self.agent_id, min_confidence]

        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.fetchall(query, tuple(params))

        insights = []
        for row in rows:
            insights.append({
                "id": row[0],
                "type": row[1],
                "title": row[2],
                "description": row[3],
                "evidence": json.loads(row[4]) if row[4] else [],
                "confidence": row[5],
                "actionable": bool(row[6]),
                "suggested_action": row[7],
                "created_at": row[8],
            })

        return insights

    async def get_behavior_rules(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get behavior rules from the database.

        Args:
            active_only: Only return active rules

        Returns:
            List of behavior rule dictionaries
        """
        if not self.db:
            return []

        query = """
            SELECT id, proposal_id, trigger_condition, action, change_type,
                   active, priority, created_at
            FROM behavior_rules
            WHERE agent_id = ?
        """
        params = [self.agent_id]

        if active_only:
            query += " AND active = 1"

        query += " ORDER BY priority DESC, created_at DESC"

        rows = await self.db.fetchall(query, tuple(params))

        rules = []
        for row in rows:
            rules.append({
                "id": row[0],
                "proposal_id": row[1],
                "trigger_condition": row[2],
                "action": row[3],
                "change_type": row[4],
                "active": bool(row[5]),
                "priority": row[6],
                "created_at": row[7],
            })

        return rules

    async def get_active_guidance(self) -> List[str]:
        """
        Get all active guidance/rules for inclusion in prompts.

        Returns:
            List of active rule actions
        """
        if not self.db:
            return []

        try:
            rows = await self.db.fetchall(
                """
                SELECT action FROM behavior_rules
                WHERE agent_id = ? AND active = 1
                ORDER BY priority DESC
                """,
                (self.agent_id,),
            )

            return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"Failed to get active guidance: {e}")
            return []

    async def get_insight_by_id(self, insight_id: str) -> Optional[Insight]:
        """
        Get a specific insight by ID.

        Args:
            insight_id: The insight ID to retrieve

        Returns:
            Insight object if found, None otherwise
        """
        if not self.db:
            return None

        try:
            rows = await self.db.fetchall(
                """
                SELECT id, type, title, description, evidence, confidence,
                       actionable, suggested_action, created_at
                FROM reflection_insights
                WHERE id = ? AND agent_id = ?
                """,
                (insight_id, self.agent_id),
            )

            if not rows:
                return None

            row = rows[0]
            return Insight(
                id=row[0],
                type=InsightType(row[1]),
                title=row[2],
                description=row[3],
                evidence=json.loads(row[4]) if row[4] else [],
                confidence=row[5],
                actionable=bool(row[6]),
                suggested_action=row[7],
                created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to get insight by ID {insight_id}: {e}")
            return None

    async def get_proposal_by_id(self, proposal_id: str) -> Optional[ImprovementProposal]:
        """Get a specific improvement proposal by ID.

        Returns None if the proposal doesn't exist or belongs to a
        different agent. Used by the create-ticket path to accept
        either an insight ID or a proposal ID — the user-facing
        ``propose_improvement`` returns proposal IDs, so without this
        the next-step ``create_improvement_ticket(proposal_id)``
        couldn't find anything (the symptom Nellie hit).
        """
        if not self.db:
            return None

        try:
            rows = await self.db.fetchall(
                """
                SELECT id, insight_id, title, description, change_type,
                       proposed_change, requires_approval, approved,
                       rejection_reason, approved_at, approved_by,
                       created_at, applied_at
                FROM improvement_proposals
                WHERE id = ? AND agent_id = ?
                """,
                (proposal_id, self.agent_id),
            )

            if not rows:
                return None

            row = rows[0]
            return ImprovementProposal(
                id=row[0],
                insight_id=row[1],
                title=row[2],
                description=row[3],
                change_type=ChangeType(row[4]),
                proposed_change=row[5],
                requires_approval=bool(row[6]),
                approved=bool(row[7]),
                rejection_reason=row[8],
                approved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                approved_by=row[10],
                created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
                applied_at=datetime.fromisoformat(row[12]) if row[12] else None,
            )
        except Exception as e:
            logger.error(f"Failed to get proposal by ID {proposal_id}: {e}")
            return None

    async def get_actionable_insights(
        self,
        from_session_id: str = None,
        limit: int = 20,
    ) -> List[Insight]:
        """
        Get actionable insights for self-model updates.

        Args:
            from_session_id: Optional session ID to filter by
            limit: Maximum number of insights to return

        Returns:
            List of Insight objects
        """
        if not self.db:
            return []

        try:
            query = """
                SELECT id, type, title, description, evidence, confidence,
                       actionable, suggested_action, created_at
                FROM reflection_insights
                WHERE agent_id = ? AND actionable = 1
            """
            params = [self.agent_id]

            if from_session_id:
                query += " AND session_id = ?"
                params.append(from_session_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = await self.db.fetchall(query, tuple(params))

            insights = []
            for row in rows:
                insights.append(Insight(
                    id=row[0],
                    type=InsightType(row[1]),
                    title=row[2],
                    description=row[3],
                    evidence=json.loads(row[4]) if row[4] else [],
                    confidence=row[5],
                    actionable=bool(row[6]),
                    suggested_action=row[7],
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                ))

            return insights

        except Exception as e:
            logger.error(f"Failed to get actionable insights: {e}")
            return []