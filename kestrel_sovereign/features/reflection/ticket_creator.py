"""
Ticket Creator for Reflection Feature.

Creates GitHub issues from agent insights with constitutional oversight.
All ticket creation requires explicit approval via the SecurityFeature's
approval queue.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ImprovementProposal  # noqa: F401
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from .models import Insight

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class GitHubIssueClient(Protocol):
    """Subset of the GitHub feature client used by reflection tickets."""

    _configured: bool

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: list[str],
    ) -> dict:
        ...


class TicketCreator:
    """Creates GitHub issues with constitutional oversight.

    This class handles the creation of GitHub issues from agent insights.
    All issue creation requires constitutional approval before execution.

    The target repository is configured via GITHUB_SELF_REPO environment
    variable, defaulting to "KestrelSovereignAI/kestrel-sovereign".
    """

    DEFAULT_REPO = "KestrelSovereignAI/kestrel-sovereign"

    def __init__(self, github_client: GitHubIssueClient):
        """Initialize the ticket creator.

        Args:
            github_client: Configured GitHubClient instance

        Raises:
            ConfigurationError: If GITHUB_PAT/GITHUB_TOKEN not set or
                              GitHub client not properly configured
        """
        # FAIL FAST - no fallbacks
        if not os.environ.get("GITHUB_PAT") and not os.environ.get("GITHUB_TOKEN"):
            raise ConfigurationError(
                "GITHUB_PAT or GITHUB_TOKEN environment variable required for ticket creation"
            )
        if not github_client._configured:
            raise ConfigurationError(
                "GitHub client not configured - check API token"
            )

        self.github = github_client
        self.repo = os.environ.get("GITHUB_SELF_REPO", self.DEFAULT_REPO)

    async def create_ticket_from_insight(
        self,
        insight: "Insight",
        security_feature: "SecurityFeature",
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Create a GitHub issue from an insight with constitutional approval.

        This method:
        1. Builds the issue content from the insight
        2. Requests constitutional approval via the security feature
        3. If approved, creates the GitHub issue
        4. Returns the issue URL or None if not approved

        Args:
            insight: The Insight to create a ticket for
            security_feature: SecurityFeature instance for approval
            timeout: Approval timeout in seconds (default 5 minutes)

        Returns:
            Issue URL if created, None if not approved or failed
        """
        # Build ticket content
        title = f"[Agent Insight] {insight.title}"
        body = self._build_ticket_body(insight)
        labels = self._suggest_labels(insight)

        # Request constitutional approval
        logger.info(f"Requesting approval for ticket: {title}")

        approval_request = {
            "feature_name": "reflection",
            "tool_name": "create_github_issue",
            "tool_args": {
                "title": insight.title,
                "type": insight.type.value,
                "labels": labels,
                "confidence": insight.confidence,
            },
        }

        try:
            approved, approval_type = await security_feature.approval_queue.request_approval(
                **approval_request,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            return None

        if not approved:
            logger.info(f"Ticket creation not approved: {title}")
            return None

        # Create the issue
        try:
            result = await self.github.create_issue(
                repo=self.repo,
                title=title,
                body=body,
                labels=labels,
            )
            issue_url = result.get("html_url")
            logger.info(f"Created ticket: {issue_url}")
            return issue_url

        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            raise

    async def create_ticket_from_proposal(
        self,
        proposal: "ImprovementProposal",
        security_feature: "SecurityFeature",
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Create a GitHub issue from an approved improvement proposal.

        Mirrors ``create_ticket_from_insight`` but uses proposal
        fields directly. The two paths share an output shape so the
        caller (``create_improvement_ticket``) can dispatch by
        whichever object the user passed in.

        Note: this approval is the *external write* gate (publishing
        to GitHub), distinct from the proposal's earlier
        *self-modification* approval. Passing an already-approved
        proposal still asks the user before posting — that's
        deliberate. See Nellie's review notes on separating proposal
        approval from external-write approval.
        """
        title = f"[Agent Proposal] {proposal.title}"
        body = self._build_ticket_body_from_proposal(proposal)
        labels = self._suggest_labels_from_proposal(proposal)

        logger.info(f"Requesting external-write approval for ticket: {title}")
        approval_request = {
            "feature_name": "reflection",
            "tool_name": "create_github_issue",
            "tool_args": {
                "title": proposal.title,
                "source": "proposal",
                "proposal_id": proposal.id,
                "change_type": proposal.change_type.value,
                "labels": labels,
                "previously_approved_for_self_modify": bool(proposal.approved),
            },
        }

        try:
            approved, _approval_type = await security_feature.approval_queue.request_approval(
                **approval_request,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Approval request failed: {e}")
            return None

        if not approved:
            logger.info(f"Ticket creation not approved: {title}")
            return None

        try:
            result = await self.github.create_issue(
                repo=self.repo,
                title=title,
                body=body,
                labels=labels,
            )
            issue_url = result.get("html_url")
            logger.info(f"Created ticket from proposal: {issue_url}")
            return issue_url
        except Exception as e:
            logger.error(f"Failed to create GitHub issue from proposal: {e}")
            raise

    def _build_ticket_body_from_proposal(self, proposal: "ImprovementProposal") -> str:
        """Markdown body derived from an ImprovementProposal."""
        sections = []
        sections.append(f"## {proposal.change_type.value.replace('_', ' ').title()} Proposal")
        sections.append("")
        sections.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
        sections.append(f"**Proposal ID:** `{proposal.id}`")
        if proposal.insight_id:
            sections.append(f"**Source Insight:** `{proposal.insight_id}`")
        sections.append(
            f"**Status:** "
            f"{'approved for self-modification' if proposal.approved else 'awaiting approval'}"
        )
        if proposal.approved_at:
            sections.append(f"**Approved At:** {proposal.approved_at.isoformat()}Z")
        sections.append("")

        sections.append("## Description")
        sections.append("")
        sections.append(proposal.description)
        sections.append("")

        sections.append("## Proposed Change")
        sections.append("")
        sections.append("```")
        sections.append(proposal.proposed_change)
        sections.append("```")
        sections.append("")

        sections.append("---")
        sections.append("*This issue was created from an approved Kestrel agent proposal.*")
        sections.append(
            "*External-write approval was obtained separately from the "
            "self-modification approval — see proposal record for details.*"
        )
        return "\n".join(sections)

    def _suggest_labels_from_proposal(self, proposal: "ImprovementProposal") -> list[str]:
        """Labels for proposal-sourced issues."""
        labels = ["agent-proposal", "self-improvement"]
        ct = proposal.change_type.value
        labels.append(f"change-type:{ct.replace('_', '-')}")
        if proposal.approved:
            labels.append("self-modify-approved")
        return labels

    def _build_ticket_body(self, insight: "Insight") -> str:
        """Build the GitHub issue body from an insight.

        Args:
            insight: The insight to convert to issue body

        Returns:
            Markdown-formatted issue body
        """
        sections = []

        # Header with metadata
        sections.append(f"## {insight.type.value.title()} Insight")
        sections.append("")
        sections.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
        sections.append(f"**Confidence:** {insight.confidence:.0%}")
        sections.append(f"**Actionable:** {'Yes' if insight.actionable else 'No'}")
        sections.append("")

        # Description
        sections.append("## Description")
        sections.append("")
        sections.append(insight.description)
        sections.append("")

        # Evidence if available
        if insight.evidence:
            sections.append("## Evidence")
            sections.append("")
            for i, evidence in enumerate(insight.evidence[:5], 1):
                sections.append(f"{i}. `{evidence}`")
            if len(insight.evidence) > 5:
                sections.append(f"... and {len(insight.evidence) - 5} more")
            sections.append("")

        # Suggested action if available
        if insight.suggested_action:
            sections.append("## Suggested Action")
            sections.append("")
            sections.append(insight.suggested_action)
            sections.append("")

        # Footer
        sections.append("---")
        sections.append("*This issue was automatically created by the Kestrel Agent's reflection system.*")
        sections.append("*Constitutional approval was obtained before creation.*")

        return "\n".join(sections)

    def _suggest_labels(self, insight: "Insight") -> list[str]:
        """Suggest GitHub labels based on insight type.

        Args:
            insight: The insight to suggest labels for

        Returns:
            List of label names
        """
        labels = ["agent-insight"]

        # Map insight types to labels
        type_labels = {
            "pattern": "pattern",
            "success": "documentation",
            "failure": "bug",
            "improvement": "enhancement",
            "anomaly": "investigation",
        }

        insight_type = insight.type.value.lower()
        if insight_type in type_labels:
            labels.append(type_labels[insight_type])

        # Add actionable label if applicable
        if insight.actionable:
            labels.append("actionable")

        # Add priority based on confidence
        if insight.confidence >= 0.9:
            labels.append("high-confidence")
        elif insight.confidence < 0.5:
            labels.append("low-confidence")

        return labels


async def create_ticket_creator(github_client: GitHubIssueClient) -> TicketCreator:
    """Factory function to create a TicketCreator.

    Args:
        github_client: Configured GitHubClient

    Returns:
        Initialized TicketCreator

    Raises:
        ConfigurationError: If required configuration is missing
    """
    return TicketCreator(github_client)
