"""
GitHub ticket creation handler for reflection feature.

Manages the creation of GitHub issues from actionable insights.
"""

import logging
from typing import Dict, Any, Optional

from kestrel_sdk.config.constants import APPROVAL_TIMEOUT_DEFAULT

from .models import Insight
from .db_helpers import ReflectionDatabaseHelper

logger = logging.getLogger(__name__)


class TicketHandler:
    """Handles GitHub ticket creation from insights."""

    def __init__(
        self,
        ticket_creator,
        economic_gate,
        db_helper: ReflectionDatabaseHelper,
        agent,
    ):
        """
        Initialize the ticket handler.

        Args:
            ticket_creator: TicketCreator instance
            economic_gate: EconomicGate for access control
            db_helper: Database helper for insight retrieval
            agent: Agent instance for feature access
        """
        self.ticket_creator = ticket_creator
        self.economic_gate = economic_gate
        self.db_helper = db_helper
        self.agent = agent

    async def create_improvement_ticket(self, insight_id: str) -> Dict[str, Any]:
        """
        Create a GitHub issue from an actionable insight.

        This requires:
        - Economic eligibility (paid tier or revenue share)
        - Constitutional approval before creation
        - GITHUB_PAT environment variable configured

        Args:
            insight_id: ID of the insight to create a ticket for

        Returns:
            Result including GitHub issue URL if created
        """
        # Check ticket creator availability
        if not self.ticket_creator:
            return {
                "success": False,
                "error": "Ticket creator not available - check GITHUB_PAT configuration",
            }

        # Check economic eligibility
        if self.economic_gate and not self.economic_gate.can_create_tickets():
            return {
                "success": False,
                "error": "Ticket creation requires paid tier or revenue share agreement",
            }

        # Get the insight from database
        if not self.db_helper:
            return {"success": False, "error": "Database not available"}

        try:
            insight = await self.db_helper.get_insight_by_id(insight_id)
            if not insight:
                return {"success": False, "error": f"Insight {insight_id} not found"}

            if not insight.actionable:
                return {
                    "success": False,
                    "error": "Insight is not marked as actionable",
                }

            # Get security feature for approval
            security = self._get_security_feature()
            if not security:
                return {
                    "success": False,
                    "error": "Security feature not available for constitutional approval",
                }

            # Create the ticket with constitutional approval
            issue_url = await self.ticket_creator.create_ticket_from_insight(
                insight=insight,
                security_feature=security,
                timeout=APPROVAL_TIMEOUT_DEFAULT,
            )

            if issue_url:
                return {
                    "success": True,
                    "issue_url": issue_url,
                    "insight_id": insight_id,
                }
            else:
                return {
                    "success": False,
                    "error": "Ticket creation not approved or failed",
                    "insight_id": insight_id,
                }

        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            return {"success": False, "error": str(e)}

    def _get_security_feature(self):
        """Get the security feature from the agent."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get("security")
        return None