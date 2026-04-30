"""
GitHub ticket creation handler for reflection feature.

Manages the creation of GitHub issues from actionable insights.
"""

import logging
from typing import Dict, Any, Optional

from kestrel_sovereign.kestrel_config.constants import APPROVAL_TIMEOUT_DEFAULT

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
        """Create a GitHub issue from an actionable insight OR an
        approved improvement proposal.

        Accepts either an insight ID or a proposal ID — the parameter
        is named ``insight_id`` for backwards compat, but a proposal
        ID resolves through ``get_proposal_by_id`` and routes to the
        proposal-shaped ticket builder. This closes the seam Nellie
        flagged: ``propose_improvement`` returns proposal IDs, but
        the next step expected insight IDs and silently failed.

        State on success:
            ``{success, issue_url, source ('insight' | 'proposal'),
             insight_id | proposal_id, state}``
        where ``state`` is one of ``"ticket_created"``.

        State on failure:
            ``{success, error, state, ...}`` where ``state`` describes
        the gate that blocked: ``"ticket_creator_unavailable"``,
        ``"economic_gate_blocked"``, ``"db_unavailable"``,
        ``"not_found"``, ``"not_actionable"``,
        ``"security_unavailable"``, ``"approval_denied_or_failed"``.

        Requires: economic eligibility (paid tier or revenue share),
        external-write approval, and ``GITHUB_PAT``/``GITHUB_TOKEN``.
        """
        if not self.ticket_creator:
            return {
                "success": False,
                "state": "ticket_creator_unavailable",
                "error": "Ticket creator not available - check GITHUB_PAT configuration",
            }

        if self.economic_gate and not self.economic_gate.can_create_tickets():
            return {
                "success": False,
                "state": "economic_gate_blocked",
                "error": "Ticket creation requires paid tier or revenue share agreement",
            }

        if not self.db_helper:
            return {
                "success": False,
                "state": "db_unavailable",
                "error": "Database not available",
            }

        try:
            insight = await self.db_helper.get_insight_by_id(insight_id)
            if insight is not None:
                return await self._ticket_from_insight(insight)

            # Fall back to proposal lookup — the user-facing
            # ``propose_improvement`` returns proposal IDs and Nellie
            # was reasonably feeding those straight into create-ticket.
            proposal = await self.db_helper.get_proposal_by_id(insight_id)
            if proposal is not None:
                return await self._ticket_from_proposal(proposal)

            return {
                "success": False,
                "state": "not_found",
                "error": (
                    f"No insight or proposal with id {insight_id!r} "
                    "found for this agent"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to create ticket: {e}", exc_info=True)
            return {
                "success": False,
                "state": "error",
                "error": str(e),
            }

    async def _ticket_from_insight(self, insight) -> Dict[str, Any]:
        if not insight.actionable:
            return {
                "success": False,
                "state": "not_actionable",
                "error": "Insight is not marked as actionable",
                "insight_id": insight.id,
            }

        security = self._get_security_feature()
        if not security:
            return {
                "success": False,
                "state": "security_unavailable",
                "error": "Security feature not available for constitutional approval",
                "insight_id": insight.id,
            }

        issue_url = await self.ticket_creator.create_ticket_from_insight(
            insight=insight,
            security_feature=security,
            timeout=APPROVAL_TIMEOUT_DEFAULT,
        )
        if issue_url:
            return {
                "success": True,
                "state": "ticket_created",
                "source": "insight",
                "issue_url": issue_url,
                "insight_id": insight.id,
            }
        return {
            "success": False,
            "state": "approval_denied_or_failed",
            "error": "Ticket creation not approved or failed",
            "insight_id": insight.id,
        }

    async def _ticket_from_proposal(self, proposal) -> Dict[str, Any]:
        if not proposal.approved:
            return {
                "success": False,
                "state": "proposal_not_approved",
                "error": (
                    "Proposal must be approved (via propose_improvement) "
                    "before its ticket can be filed"
                ),
                "proposal_id": proposal.id,
            }

        security = self._get_security_feature()
        if not security:
            return {
                "success": False,
                "state": "security_unavailable",
                "error": "Security feature not available for constitutional approval",
                "proposal_id": proposal.id,
            }

        issue_url = await self.ticket_creator.create_ticket_from_proposal(
            proposal=proposal,
            security_feature=security,
            timeout=APPROVAL_TIMEOUT_DEFAULT,
        )
        if issue_url:
            return {
                "success": True,
                "state": "ticket_created",
                "source": "proposal",
                "issue_url": issue_url,
                "proposal_id": proposal.id,
            }
        return {
            "success": False,
            "state": "approval_denied_or_failed",
            "error": "Ticket creation not approved or failed",
            "proposal_id": proposal.id,
        }

    def _get_security_feature(self):
        """Get the security feature from the agent."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get("security")
        return None