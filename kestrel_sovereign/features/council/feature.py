"""
Constitutional Council Feature

Exposes council functionality as agent tools for Kestrel.
"""

import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool, ToolCategory
from .models import (
    CouncilConfig,
    CouncilMember,
    ConsensusRule,
    SessionOutcome,
)
from .evidence import compile_evidence, compile_emma_genesis_evidence
from .deliberation import convene_council, apply_human_override
from .storage import CouncilStorage, get_storage

logger = logging.getLogger(__name__)

# Default config path (individual file for backward compat)
CONFIG_PATH = Path("council_config.toml")

# Unified config path (preferred)
UNIFIED_CONFIG_PATH = Path("kestrel.toml")


class CouncilFeature(Feature):
    """
    Constitutional Council - Multi-model deliberation for major decisions.

    This feature allows the sovereign to convene a council of foundation models
    to deliberate on important decisions before taking irreversible actions.
    """

    def __init__(self, agent):
        super().__init__(agent)
        self.config: Optional[CouncilConfig] = None
        self.storage: Optional[CouncilStorage] = None

    @property
    def tool_description(self) -> str:
        return (
            "Convene a Constitutional Council of foundation models to deliberate "
            "on major decisions. Use for irreversible actions like creating agents, "
            "constitutional amendments, or key changes."
        )

    async def initialize(self):
        """Initialize the council feature."""
        self.storage = get_storage()

        config_source = None
        config_data = None

        # Try unified config first
        if UNIFIED_CONFIG_PATH.exists():
            try:
                with open(UNIFIED_CONFIG_PATH, "rb") as f:
                    unified_data = tomllib.load(f)
                if "council" in unified_data:
                    config_data = unified_data.get("council", {})
                    config_source = "kestrel.toml"
                    logger.debug("Loading council config from unified kestrel.toml")
            except Exception as e:
                logger.warning(f"Could not load council config from unified file: {e}")

        # Fall back to individual config file
        if not config_data and CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "rb") as f:
                    data = tomllib.load(f)
                config_data = data.get("council", {})
                config_source = str(CONFIG_PATH)

                # Log deprecation warning if unified config exists
                if UNIFIED_CONFIG_PATH.exists():
                    logger.warning(
                        "DEPRECATION: Loading from 'council_config.toml' directly. "
                        "Consider migrating to unified 'kestrel.toml' configuration. "
                        "Individual config files will be removed in a future version."
                    )
            except Exception as e:
                logger.warning(f"Could not load council config: {e}")

        # Parse config if found
        if config_data:
            try:
                self.config = CouncilConfig.from_dict(config_data)
                logger.info(
                    f"Loaded council config with {len(self.config.members)} members from {config_source}"
                )
            except Exception as e:
                logger.warning(f"Could not parse council config: {e}")
        else:
            logger.info("No council config found, will require explicit members")

    async def shutdown(self):
        """Cleanup resources."""
        pass

    @tool(
        "council_convene",
        "Convene the Constitutional Council to deliberate on a question. "
        "Returns the session outcome (APPROVED/REJECTED/DEADLOCK).",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-convene"
    )
    async def convene(
        self,
        question: str,
        target: str = "general",
        max_rounds: int = 3,
    ) -> str:
        """
        Convene the Constitutional Council for deliberation.

        Args:
            question: The question to deliberate on
            target: Evidence target (e.g., 'emma_genesis', 'general')
            max_rounds: Maximum deliberation rounds (default 3)

        Returns:
            Session summary with outcome
        """
        if not self.config or not self.config.members:
            return (
                "Error: No council members configured. "
                "Create council_config.toml or provide members explicitly."
            )

        if len(self.config.members) < self.config.min_members:
            return (
                f"Error: Council requires at least {self.config.min_members} members, "
                f"but only {len(self.config.members)} configured."
            )

        # Compile evidence
        if target == "emma_genesis":
            evidence = await compile_emma_genesis_evidence()
        else:
            evidence = await compile_evidence(target=target)

        # Run deliberation
        session = await convene_council(
            question=question,
            evidence=evidence,
            members=self.config.members,
            max_rounds=min(max_rounds, self.config.max_rounds),
            consensus_rule=self.config.consensus_rule,
        )

        # Save session
        await self.storage.save_session(session)

        # Format response
        outcome_emoji = {
            SessionOutcome.APPROVED: "✅",
            SessionOutcome.REJECTED: "❌",
            SessionOutcome.DEADLOCK: "⚠️",
            SessionOutcome.PENDING: "⏳",
        }.get(session.outcome, "❓")

        response_parts = [
            f"# Council Session: {session.id}",
            f"## Outcome: {outcome_emoji} {session.outcome.value}",
            "",
            f"**Question:** {question}",
            f"**Members:** {len(session.members)}",
            f"**Rounds:** {len(session.rounds)}",
            "",
            "## Verdicts",
        ]

        for verdict in session.verdicts:
            v_emoji = {"APPROVE": "✅", "REJECT": "❌", "ABSTAIN": "⚪"}.get(
                verdict.decision.value, "?"
            )
            response_parts.append(
                f"\n### {v_emoji} {verdict.member_name}: {verdict.decision.value} "
                f"({verdict.confidence:.0%} confidence)"
            )
            response_parts.append(f"\n{verdict.reasoning[:500]}...")
            if verdict.concerns:
                response_parts.append("\n**Concerns:**")
                for c in verdict.concerns[:3]:
                    response_parts.append(f"- {c}")

        response_parts.append(f"\n\n*Full transcript saved to: data/council_sessions/{session.id}.md*")

        return "\n".join(response_parts)

    @tool(
        "council_status",
        "View status of council sessions, including recent decisions.",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-status"
    )
    async def status(
        self,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        View council session status.

        Args:
            session_id: Specific session ID to view (optional)
            limit: Number of recent sessions to list (default 5)

        Returns:
            Session details or list of recent sessions
        """
        if session_id:
            session = await self.storage.load_session(session_id)
            if not session:
                return f"Session {session_id} not found."
            return session.to_transcript()

        sessions = await self.storage.list_sessions(limit=limit)
        if not sessions:
            return "No council sessions found."

        lines = ["# Recent Council Sessions", ""]
        for s in sessions:
            emoji = {"APPROVED": "✅", "REJECTED": "❌", "DEADLOCK": "⚠️"}.get(
                s["outcome"], "?"
            )
            lines.append(
                f"- {emoji} **{s['id'][:8]}...** {s['question'][:50]}... "
                f"({s['outcome']})"
            )

        return "\n".join(lines)

    @tool(
        "council_override",
        "Apply a human override to a council session (Sovereign authority).",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-override"
    )
    async def override(
        self,
        session_id: str,
        decision: str,
        reason: str,
    ) -> str:
        """
        Apply a human override to a council decision.

        Args:
            session_id: Session ID to override
            decision: APPROVE or REJECT
            reason: Reason for the override

        Returns:
            Confirmation of override
        """
        session = await self.storage.load_session(session_id)
        if not session:
            return f"Session {session_id} not found."

        if decision.upper() not in ["APPROVE", "REJECT"]:
            return "Decision must be APPROVE or REJECT."

        apply_human_override(session, decision.upper(), reason)
        await self.storage.save_session(session)

        return (
            f"Human override applied to session {session_id}.\n"
            f"New outcome: {session.outcome.value}\n"
            f"Reason: {reason}"
        )

    @tool(
        "council_members",
        "List configured council members.",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-members"
    )
    async def list_members(self) -> str:
        """
        List the configured council members.

        Returns:
            List of council members with their roles
        """
        if not self.config or not self.config.members:
            return (
                "No council members configured.\n\n"
                "Create council_config.toml with:\n"
                "```toml\n"
                "[council]\n"
                "min_members = 3\n"
                "consensus_rule = \"unanimous\"\n\n"
                "[[council.members]]\n"
                "name = \"Claude\"\n"
                "provider = \"anthropic\"\n"
                "model = \"claude-opus-4-5-20251101\"\n"
                "role = \"constitutional_reviewer\"\n"
                "```"
            )

        lines = [
            "# Constitutional Council Members",
            f"Consensus Rule: {self.config.consensus_rule.value}",
            f"Minimum Members: {self.config.min_members}",
            f"Max Rounds: {self.config.max_rounds}",
            "",
            "## Members",
        ]

        for member in self.config.members:
            lines.append(
                f"\n### {member.name}\n"
                f"- **Provider:** {member.provider}\n"
                f"- **Model:** {member.model}\n"
                f"- **Role:** {member.role}"
            )

        return "\n".join(lines)

    @tool(
        "council_evidence",
        "Preview the evidence package that would be presented to the council.",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-evidence"
    )
    async def preview_evidence(
        self,
        target: str = "general",
    ) -> str:
        """
        Preview the evidence package for a council session.

        Args:
            target: Evidence target (e.g., 'emma_genesis', 'general')

        Returns:
            Formatted evidence preview
        """
        if target == "emma_genesis":
            evidence = await compile_emma_genesis_evidence()
        else:
            evidence = await compile_evidence(target=target)

        lines = [
            "# Evidence Package Preview",
            f"Target: {evidence.target}",
            f"Compiled: {evidence.compiled_at.isoformat()}",
            f"Content Hash: {evidence.content_hash()}",
            "",
            "## Summary",
            f"- Code changes: {len(evidence.code_changes)} items",
            f"- Tests: {evidence.test_passed}/{evidence.test_count} passing",
            f"- Risks: {len(evidence.risks)} identified",
            f"- Architecture docs: {len(evidence.architecture_docs)} files",
            f"- Previous decisions: {len(evidence.previous_decisions)}",
            "",
        ]

        if evidence.risks:
            lines.append("## Risks")
            for risk in evidence.risks[:5]:
                lines.append(f"- {risk}")
            lines.append("")

        lines.append("---")
        lines.append("*This is a preview. Use council_convene to run a full session.*")

        return "\n".join(lines)
