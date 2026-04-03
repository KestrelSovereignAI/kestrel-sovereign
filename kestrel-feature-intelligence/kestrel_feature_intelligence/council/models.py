"""
Constitutional Council Data Models

Core dataclasses for the multi-model deliberation system.
Model-agnostic design - any foundation model can participate.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import hashlib
import json
import uuid


class ConsensusRule(str, Enum):
    """Rules for determining council consensus."""
    UNANIMOUS = "unanimous"       # All must approve
    SUPERMAJORITY = "supermajority"  # 75% must approve
    MAJORITY = "majority"         # 50%+1 must approve
    QUORUM = "quorum"             # At least min_members must approve


class Decision(str, Enum):
    """Possible verdict decisions."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class SessionOutcome(str, Enum):
    """Possible session outcomes."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEADLOCK = "DEADLOCK"
    PENDING = "PENDING"


@dataclass
class CouncilMember:
    """
    A member of the Constitutional Council.

    Model-agnostic: any provider/model combination works.
    """
    name: str               # Display name (e.g., "Claude", "GPT", "Gemini")
    provider: str           # Provider ID (e.g., "anthropic", "openai", "google")
    model: str              # Full model ID from provider
    role: str               # Role in deliberation (e.g., "constitutional_reviewer")

    def __post_init__(self):
        """Validate member configuration."""
        valid_providers = [
            "anthropic", "openai", "google", "vertex_ai",
            "ollama", "xai", "groq"
        ]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Valid providers: {valid_providers}"
            )


@dataclass
class Evidence:
    """
    Evidence package presented to the council for deliberation.

    Contains all relevant information for the decision at hand.
    """
    # Code and testing
    code_changes: List[str] = field(default_factory=list)  # Git diffs
    test_results: Dict[str, Any] = field(default_factory=dict)  # pytest output
    test_count: int = 0
    test_passed: int = 0
    test_failed: int = 0

    # Security and architecture
    security_assessment: str = ""
    architecture_docs: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # History
    previous_decisions: List[str] = field(default_factory=list)
    related_commits: List[str] = field(default_factory=list)

    # Metadata
    target: str = ""  # What this evidence is for (e.g., "emma_genesis")
    compiled_at: datetime = field(default_factory=datetime.utcnow)
    source_files: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert evidence to a prompt-friendly format."""
        sections = []

        if self.target:
            sections.append(f"# Evidence Package: {self.target}")
            sections.append(f"Compiled: {self.compiled_at.isoformat()}")
            sections.append("")

        if self.code_changes:
            sections.append("## Code Changes")
            for change in self.code_changes[:10]:  # Limit for context
                sections.append(f"```\n{change}\n```")
            sections.append("")

        if self.test_results:
            sections.append("## Test Results")
            sections.append(f"- Total: {self.test_count}")
            sections.append(f"- Passed: {self.test_passed}")
            sections.append(f"- Failed: {self.test_failed}")
            if self.test_results.get("summary"):
                sections.append(f"\n{self.test_results['summary']}")
            sections.append("")

        if self.security_assessment:
            sections.append("## Security Assessment")
            sections.append(self.security_assessment)
            sections.append("")

        if self.risks:
            sections.append("## Known Risks")
            for risk in self.risks:
                sections.append(f"- {risk}")
            sections.append("")

        if self.architecture_docs:
            sections.append("## Architecture Documentation")
            for doc in self.architecture_docs[:5]:  # Limit
                sections.append(doc)
            sections.append("")

        if self.previous_decisions:
            sections.append("## Previous Council Decisions")
            for decision in self.previous_decisions[-3:]:  # Last 3
                sections.append(f"- {decision}")
            sections.append("")

        return "\n".join(sections)

    def content_hash(self) -> str:
        """Generate a content-addressable hash of the evidence."""
        content = json.dumps({
            "target": self.target,
            "code_changes": self.code_changes,
            "test_count": self.test_count,
            "risks": self.risks,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Verdict:
    """
    A council member's verdict on the question at hand.
    """
    member_name: str                           # Who issued the verdict
    model: str                                  # Full model ID used
    decision: Decision                          # APPROVE, REJECT, ABSTAIN
    confidence: float                           # 0.0 - 1.0
    reasoning: str                              # Detailed explanation
    concerns: List[str] = field(default_factory=list)      # Remaining concerns
    conditions: List[str] = field(default_factory=list)    # "Approve IF..."
    dissent: Optional[str] = None               # If outvoted, record dissent
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate verdict."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if isinstance(self.decision, str):
            self.decision = Decision(self.decision)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "member_name": self.member_name,
            "model": self.model,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "concerns": self.concerns,
            "conditions": self.conditions,
            "dissent": self.dissent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenUsage:
    """Token usage for a single model invocation."""
    member_name: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    round_number: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_name": self.member_name,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "round_number": self.round_number,
        }


@dataclass
class DeliberationMessage:
    """A single message in the deliberation transcript."""
    member_name: str
    model: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_name": self.member_name,
            "model": self.model,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DeliberationRound:
    """
    A round of deliberation where all council members respond.
    """
    round_number: int
    messages: List[DeliberationMessage] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_message(
        self,
        member_name: str,
        model: str,
        content: str
    ) -> DeliberationMessage:
        """Add a message to this round."""
        msg = DeliberationMessage(
            member_name=member_name,
            model=model,
            content=content
        )
        self.messages.append(msg)
        return msg

    def to_transcript(self) -> str:
        """Convert round to readable transcript."""
        lines = [f"### Round {self.round_number}"]
        for msg in self.messages:
            lines.append(f"\n**{msg.member_name}** ({msg.model}):")
            lines.append(msg.content)
        return "\n".join(lines)


@dataclass
class CouncilSession:
    """
    A complete council deliberation session.

    Tracks the question, evidence, all rounds of deliberation,
    final verdicts, outcome, and token usage.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    evidence: Optional[Evidence] = None
    members: List[CouncilMember] = field(default_factory=list)
    rounds: List[DeliberationRound] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    token_usage: List[TokenUsage] = field(default_factory=list)
    outcome: SessionOutcome = SessionOutcome.PENDING
    consensus_rule: ConsensusRule = ConsensusRule.UNANIMOUS
    human_override: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def add_token_usage(
        self,
        member_name: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        round_number: int
    ) -> None:
        """Record token usage for a model invocation."""
        self.token_usage.append(TokenUsage(
            member_name=member_name,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            round_number=round_number,
        ))

    def total_tokens(self) -> Dict[str, int]:
        """Get total tokens across all invocations."""
        return {
            "input": sum(u.input_tokens for u in self.token_usage),
            "output": sum(u.output_tokens for u in self.token_usage),
            "total": sum(u.total_tokens for u in self.token_usage),
        }

    def tokens_by_member(self) -> Dict[str, Dict[str, int]]:
        """Get tokens grouped by member."""
        by_member: Dict[str, Dict[str, int]] = {}
        for usage in self.token_usage:
            if usage.member_name not in by_member:
                by_member[usage.member_name] = {
                    "input": 0, "output": 0, "provider": usage.provider, "model": usage.model
                }
            by_member[usage.member_name]["input"] += usage.input_tokens
            by_member[usage.member_name]["output"] += usage.output_tokens
        return by_member

    def add_round(self) -> DeliberationRound:
        """Add a new deliberation round."""
        round_num = len(self.rounds) + 1
        new_round = DeliberationRound(round_number=round_num)
        self.rounds.append(new_round)
        return new_round

    def add_verdict(self, verdict: Verdict) -> None:
        """Add a verdict and check for consensus."""
        self.verdicts.append(verdict)
        self._update_outcome()

    def _update_outcome(self) -> None:
        """Update session outcome based on verdicts and consensus rule."""
        if len(self.verdicts) < len(self.members):
            self.outcome = SessionOutcome.PENDING
            return

        approve_count = sum(
            1 for v in self.verdicts if v.decision == Decision.APPROVE
        )
        reject_count = sum(
            1 for v in self.verdicts if v.decision == Decision.REJECT
        )
        total = len(self.verdicts)

        if self.consensus_rule == ConsensusRule.UNANIMOUS:
            if approve_count == total:
                self.outcome = SessionOutcome.APPROVED
            elif reject_count > 0:
                self.outcome = SessionOutcome.REJECTED
            else:
                self.outcome = SessionOutcome.DEADLOCK

        elif self.consensus_rule == ConsensusRule.SUPERMAJORITY:
            if approve_count >= total * 0.75:
                self.outcome = SessionOutcome.APPROVED
            elif reject_count > total * 0.25:
                self.outcome = SessionOutcome.REJECTED
            else:
                self.outcome = SessionOutcome.DEADLOCK

        elif self.consensus_rule == ConsensusRule.MAJORITY:
            if approve_count > total / 2:
                self.outcome = SessionOutcome.APPROVED
            elif reject_count >= total / 2:
                self.outcome = SessionOutcome.REJECTED
            else:
                self.outcome = SessionOutcome.DEADLOCK

        if self.outcome != SessionOutcome.PENDING:
            self.completed_at = datetime.utcnow()

    def to_transcript(self) -> str:
        """Generate full session transcript."""
        lines = [
            f"# Council Session: {self.id}",
            f"Created: {self.created_at.isoformat()}",
            f"Consensus Rule: {self.consensus_rule.value}",
            "",
            "## Question",
            self.question,
            "",
            "## Council Members",
        ]

        for member in self.members:
            lines.append(
                f"- **{member.name}** ({member.provider}/{member.model}) "
                f"- {member.role}"
            )

        lines.append("")
        lines.append("## Evidence Summary")
        if self.evidence:
            lines.append(f"Target: {self.evidence.target}")
            lines.append(f"Tests: {self.evidence.test_passed}/{self.evidence.test_count} passing")
            lines.append(f"Risks: {len(self.evidence.risks)}")

        lines.append("")
        lines.append("## Deliberation")
        for round in self.rounds:
            lines.append(round.to_transcript())

        lines.append("")
        lines.append("## Verdicts")
        for verdict in self.verdicts:
            emoji = {
                Decision.APPROVE: "✅",
                Decision.REJECT: "❌",
                Decision.ABSTAIN: "⚪"
            }.get(verdict.decision, "?")
            lines.append(
                f"\n### {emoji} {verdict.member_name}: {verdict.decision.value} "
                f"(confidence: {verdict.confidence:.0%})"
            )
            lines.append(verdict.reasoning)
            if verdict.concerns:
                lines.append("\n**Concerns:**")
                for concern in verdict.concerns:
                    lines.append(f"- {concern}")
            if verdict.conditions:
                lines.append("\n**Conditions:**")
                for condition in verdict.conditions:
                    lines.append(f"- {condition}")

        lines.append("")
        lines.append("## Outcome")
        lines.append(f"**{self.outcome.value}**")
        if self.human_override:
            lines.append(f"\nHuman Override: {self.human_override}")
        if self.completed_at:
            lines.append(f"\nCompleted: {self.completed_at.isoformat()}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "question": self.question,
            "evidence_hash": self.evidence.content_hash() if self.evidence else None,
            "members": [
                {"name": m.name, "provider": m.provider, "model": m.model, "role": m.role}
                for m in self.members
            ],
            "rounds": [
                {
                    "round_number": r.round_number,
                    "messages": [m.to_dict() for m in r.messages],
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.rounds
            ],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "outcome": self.outcome.value,
            "consensus_rule": self.consensus_rule.value,
            "human_override": self.human_override,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class CouncilConfig:
    """Configuration for the Constitutional Council."""
    min_members: int = 3
    max_rounds: int = 5
    consensus_rule: ConsensusRule = ConsensusRule.UNANIMOUS
    members: List[CouncilMember] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CouncilConfig":
        """Create config from dictionary (e.g., parsed TOML)."""
        members = []
        for m in data.get("members", []):
            members.append(CouncilMember(
                name=m["name"],
                provider=m["provider"],
                model=m["model"],
                role=m.get("role", "reviewer"),
            ))

        rule_str = data.get("consensus_rule", "unanimous")
        try:
            rule = ConsensusRule(rule_str)
        except ValueError:
            rule = ConsensusRule.UNANIMOUS

        return cls(
            min_members=data.get("min_members", 3),
            max_rounds=data.get("max_rounds", 5),
            consensus_rule=rule,
            members=members,
        )
