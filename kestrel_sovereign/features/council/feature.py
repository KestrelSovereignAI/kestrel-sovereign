"""
Constitutional Council Feature

Exposes council functionality as agent tools for Kestrel.
"""

import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool, ToolCategory
from kestrel_sdk.tools.result import ToolResult
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
    ) -> ToolResult:
        """
        Convene the Constitutional Council for deliberation.

        Args:
            question: The question to deliberate on
            target: Evidence target (e.g., 'emma_genesis', 'general')
            max_rounds: Maximum deliberation rounds (default 3)
        """
        try:
            max_rounds_val = int(max_rounds)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"max_rounds must be an integer, got {max_rounds!r}"
            )
        if max_rounds_val < 1:
            return ToolResult.failed("max_rounds must be >= 1")

        if not self.config or not self.config.members:
            return ToolResult.failed(
                "No council members configured. "
                "Create council_config.toml or provide members explicitly."
            )

        if len(self.config.members) < self.config.min_members:
            return ToolResult.failed(
                f"Council requires at least {self.config.min_members} "
                f"members, but only {len(self.config.members)} "
                "configured.",
                data={
                    "configured_members": len(self.config.members),
                    "min_required": self.config.min_members,
                },
            )

        try:
            if target == "emma_genesis":
                evidence = await compile_emma_genesis_evidence()
            else:
                evidence = await compile_evidence(target=target)

            session = await convene_council(
                question=question,
                evidence=evidence,
                members=self.config.members,
                max_rounds=min(max_rounds_val, self.config.max_rounds),
                consensus_rule=self.config.consensus_rule,
            )
            await self.storage.save_session(session)
        except Exception as e:
            logger.error(f"council_convene failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # Format response (preserve the human-readable transcript
        # for the confirmation; surface structured deliberation
        # state via data so the LLM can read who decided what
        # without parsing markdown).
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

        verdicts_data = []
        approve_count = 0
        reject_count = 0
        abstain_count = 0
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
            decision_str = verdict.decision.value
            if decision_str == "APPROVE":
                approve_count += 1
            elif decision_str == "REJECT":
                reject_count += 1
            elif decision_str == "ABSTAIN":
                abstain_count += 1
            verdicts_data.append({
                "member_name": verdict.member_name,
                "decision": decision_str,
                "confidence": verdict.confidence,
                "concerns": list(verdict.concerns or []),
            })

        response_parts.append(
            f"\n\n*Full transcript saved to: data/council_sessions/{session.id}.md*"
        )
        confirmation = "\n".join(response_parts)
        data = {
            "session_id": session.id,
            "outcome": session.outcome.value,
            "question": question,
            "target": target,
            "members": len(session.members),
            "rounds": len(session.rounds),
            "verdicts": verdicts_data,
            "approve_count": approve_count,
            "reject_count": reject_count,
            "abstain_count": abstain_count,
            "max_rounds_requested": max_rounds_val,
            "max_rounds_applied": min(max_rounds_val, self.config.max_rounds),
        }

        # Honesty: a DEADLOCK is not a tool failure (the council
        # ran, but did not reach consensus). The LLM cannot say
        # "the council approved" — surface as PARTIAL with the
        # verdict counts so the divergence speaks. PENDING is the
        # same shape (no terminal outcome).
        if session.outcome in (SessionOutcome.DEADLOCK, SessionOutcome.PENDING):
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"council did not reach a terminal decision: "
                    f"outcome={session.outcome.value} "
                    f"(approve={approve_count}, reject={reject_count}, "
                    f"abstain={abstain_count}). The question is unresolved; "
                    "use !council-override or convene again with adjusted "
                    "evidence."
                ),
                data=data,
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

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
    ) -> ToolResult:
        """
        View council session status.

        Args:
            session_id: Specific session ID to view (optional)
            limit: Number of recent sessions to list (default 5).
                   Actual count returned may be lower.
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"limit must be an integer, got {limit!r}"
            )
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        try:
            if session_id:
                session = await self.storage.load_session(session_id)
                if not session:
                    return ToolResult.failed(
                        f"Session {session_id} not found.",
                        data={"session_id": session_id},
                    )
                return ToolResult.ok(
                    confirmation=session.to_transcript(),
                    data={
                        "session_id": session.id,
                        "outcome": session.outcome.value,
                        "members": len(session.members),
                        "rounds": len(session.rounds),
                    },
                )

            sessions = await self.storage.list_sessions(limit=limit_val)
        except Exception as e:
            logger.error(f"council_status failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not sessions:
            return ToolResult.ok(
                confirmation="No council sessions found.",
                data={"count": 0, "limit_requested": limit_val, "sessions": []},
            )

        lines = ["# Recent Council Sessions", ""]
        for s in sessions:
            emoji = {"APPROVED": "✅", "REJECTED": "❌", "DEADLOCK": "⚠️"}.get(
                s["outcome"], "?"
            )
            lines.append(
                f"- {emoji} **{s['id'][:8]}...** {s['question'][:50]}... "
                f"({s['outcome']})"
            )

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(sessions),
                "limit_requested": limit_val,
                "sessions": [
                    {
                        "id": s.get("id"),
                        "question": s.get("question"),
                        "outcome": s.get("outcome"),
                    }
                    for s in sessions
                ],
            },
        )

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
    ) -> ToolResult:
        """
        Apply a human override to a council decision.

        Args:
            session_id: Session ID to override
            decision: APPROVE or REJECT
            reason: Reason for the override
        """
        if not isinstance(reason, str) or not reason.strip():
            return ToolResult.failed(
                "reason is required (council overrides leave an audit trail)"
            )

        decision_upper = decision.upper() if isinstance(decision, str) else ""
        if decision_upper not in ("APPROVE", "REJECT"):
            return ToolResult.failed(
                f"decision must be APPROVE or REJECT, got {decision!r}"
            )

        try:
            session = await self.storage.load_session(session_id)
            if not session:
                return ToolResult.failed(
                    f"Session {session_id} not found.",
                    data={"session_id": session_id},
                )
            previous_outcome = session.outcome.value
            apply_human_override(session, decision_upper, reason)
            await self.storage.save_session(session)
        except Exception as e:
            logger.error(f"council_override failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        return ToolResult.ok(
            confirmation=(
                f"Human override applied to session {session_id}: "
                f"{previous_outcome} → {session.outcome.value} "
                f"(reason: {reason})"
            ),
            data={
                "session_id": session_id,
                "previous_outcome": previous_outcome,
                "new_outcome": session.outcome.value,
                "decision": decision_upper,
                "reason": reason,
            },
        )

    @tool(
        "council_members",
        "List configured council members.",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-members"
    )
    async def list_members(self) -> ToolResult:
        """List the configured council members."""
        if not self.config or not self.config.members:
            return ToolResult.ok(
                confirmation=(
                    "No council members configured.\n\n"
                    "Create council_config.toml with:\n"
                    "```toml\n"
                    "[council]\n"
                    "min_members = 3\n"
                    "consensus_rule = \"unanimous\"\n\n"
                    "[[council.members]]\n"
                    "name = \"Claude\"\n"
                    "provider = \"anthropic\"\n"
                    "model = \"auto\"\n"
                    "role = \"constitutional_reviewer\"\n"
                    "```"
                ),
                data={"member_count": 0, "members": []},
            )

        lines = [
            "# Constitutional Council Members",
            f"Consensus Rule: {self.config.consensus_rule.value}",
            f"Minimum Members: {self.config.min_members}",
            f"Max Rounds: {self.config.max_rounds}",
            "",
            "## Members",
        ]
        members_data = []
        for member in self.config.members:
            lines.append(
                f"\n### {member.name}\n"
                f"- **Provider:** {member.provider}\n"
                f"- **Model:** {member.model}\n"
                f"- **Role:** {member.role}"
            )
            members_data.append({
                "name": member.name,
                "provider": member.provider,
                "model": member.model,
                "role": member.role,
            })

        data = {
            "member_count": len(members_data),
            "members": members_data,
            "consensus_rule": self.config.consensus_rule.value,
            "min_members": self.config.min_members,
            "max_rounds": self.config.max_rounds,
        }

        # Honesty: the council needs at least min_members configured
        # to convene. If we have fewer, list_members must surface
        # that — the LLM might otherwise list members and proceed
        # to !council-convene which will then error.
        if len(self.config.members) < self.config.min_members:
            return ToolResult.partial(
                confirmation="\n".join(lines),
                error=(
                    f"council has {len(self.config.members)} member(s) "
                    f"but requires at least {self.config.min_members} to "
                    "convene; !council-convene will refuse to run"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data=data,
        )

    @tool(
        "council_evidence",
        "Preview the evidence package that would be presented to the council.",
        category=ToolCategory.SYSTEM,
        command_prefix="!council-evidence"
    )
    async def preview_evidence(
        self,
        target: str = "general",
    ) -> ToolResult:
        """
        Preview the evidence package for a council session.

        Args:
            target: Evidence target (e.g., 'emma_genesis', 'general')
        """
        try:
            if target == "emma_genesis":
                evidence = await compile_emma_genesis_evidence()
            else:
                evidence = await compile_evidence(target=target)
        except Exception as e:
            logger.error(f"council_evidence failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

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

        data = {
            "target": evidence.target,
            "compiled_at": evidence.compiled_at.isoformat(),
            "content_hash": evidence.content_hash(),
            "code_changes_count": len(evidence.code_changes),
            "test_passed": evidence.test_passed,
            "test_count": evidence.test_count,
            "risks_count": len(evidence.risks),
            "architecture_docs_count": len(evidence.architecture_docs),
            "previous_decisions_count": len(evidence.previous_decisions),
            "risks": list(evidence.risks)[:5],
        }

        # Honesty: tests-failing in the evidence is a signal the
        # council should weigh heavily. The LLM should not produce
        # a "looks good, convene the council" framing while the
        # evidence itself shows test failures. Surface as PARTIAL.
        if evidence.test_count > 0 and evidence.test_passed < evidence.test_count:
            failed = evidence.test_count - evidence.test_passed
            return ToolResult.partial(
                confirmation="\n".join(lines),
                error=(
                    f"evidence shows {failed} of {evidence.test_count} "
                    "test(s) failing — convening the council on a red "
                    "test suite is unusual; consider stabilizing first"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data=data,
        )
