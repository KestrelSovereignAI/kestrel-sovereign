"""
Constitutional approval handling for reflection feature.

Manages approval requests and improvement application logic.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from kestrel_sovereign.kestrel_config.constants import APPROVAL_TIMEOUT_DEFAULT

from .models import ImprovementProposal, BehaviorRule, ChangeType
from .prompts import format_approval_prompt
from .db_helpers import ReflectionDatabaseHelper

logger = logging.getLogger(__name__)


class ApprovalHandler:
    """Handles constitutional approval for self-modifications."""

    def __init__(self, agent, db_helper: Optional[ReflectionDatabaseHelper] = None):
        """
        Initialize the approval handler.

        Args:
            agent: The agent instance
            db_helper: Database helper for storing behavior rules
        """
        self.agent = agent
        self.db_helper = db_helper

    async def request_approval(self, proposal: ImprovementProposal) -> bool:
        """
        Request constitutional approval for self-modification.

        Args:
            proposal: The improvement proposal to approve

        Returns:
            True if approved, False otherwise
        """
        # Check if we have access to security feature
        security = self._get_security_feature()

        if not security or not hasattr(security, 'approval_queue'):
            logger.warning("SecurityFeature not available, auto-rejecting self-modification")
            proposal.rejection_reason = "Security feature not available"
            return False

        # Build approval request context using prompt formatter
        context = format_approval_prompt(proposal)

        try:
            approved, scope = await security.approval_queue.request_approval(
                feature_name="reflection",
                tool_name="self_modify",
                tool_args={
                    "proposal_id": proposal.id,
                    "change_type": proposal.change_type.value,
                    "title": proposal.title,
                    "description": proposal.description[:500],
                },
                timeout=APPROVAL_TIMEOUT_DEFAULT,  # 5 minutes
            )

            proposal.approved = approved
            if approved:
                proposal.approved_at = datetime.utcnow()
                proposal.approved_by = "user"
            else:
                proposal.rejection_reason = "User denied self-modification"

            return approved

        except TimeoutError:
            proposal.rejection_reason = "Approval request timed out"
            return False
        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            proposal.rejection_reason = f"Approval error: {e}"
            return False

    async def apply_improvement(self, proposal: ImprovementProposal) -> None:
        """
        Apply an approved improvement.

        Args:
            proposal: The approved improvement proposal
        """
        if not proposal.approved:
            raise ValueError("Cannot apply unapproved proposal")

        if proposal.change_type == ChangeType.PROMPT:
            # Add to agent's prompt additions
            await self._add_prompt_guidance(proposal)
        else:
            # Store as behavioral rule
            await self._add_behavior_rule(proposal)

        logger.info(f"Applied improvement: {proposal.title}")

    async def _add_prompt_guidance(self, proposal: ImprovementProposal) -> None:
        """Add guidance to the agent's prompt additions."""
        # Store as a behavior rule with "always" trigger
        rule = BehaviorRule(
            id=str(uuid.uuid4()),
            proposal_id=proposal.id,
            trigger_condition="always",
            action=proposal.proposed_change,
            change_type=proposal.change_type,
            priority=10,  # Higher priority for prompt additions
        )

        if self.db_helper:
            await self.db_helper.store_behavior_rule(rule)

    async def _add_behavior_rule(self, proposal: ImprovementProposal) -> None:
        """Add a behavior rule based on the proposal."""
        # Parse trigger condition from description if available
        trigger = "contextual"  # Default trigger

        rule = BehaviorRule(
            id=str(uuid.uuid4()),
            proposal_id=proposal.id,
            trigger_condition=trigger,
            action=proposal.proposed_change,
            change_type=proposal.change_type,
            priority=5,
        )

        if self.db_helper:
            await self.db_helper.store_behavior_rule(rule)

    def _get_security_feature(self):
        """Get the security feature from the agent."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get("security")
        return None