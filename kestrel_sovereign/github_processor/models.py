"""Data models for GitHub Ticket Processor."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ProcessingStatus(Enum):
    """Status of issue processing."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    CLARIFYING = "clarifying"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class IssueComment:
    """A comment on a GitHub issue."""

    id: int
    author: str
    body: str
    created_at: datetime
    is_bot: bool = False


@dataclass
class IssueContext:
    """Context gathered from a GitHub issue for processing."""

    number: int
    title: str
    body: str
    author: str
    labels: list[str]
    assignees: list[str]
    comments: list[IssueComment]
    created_at: datetime
    updated_at: datetime
    url: str

    # Extracted context
    referenced_files: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    linked_prs: list[int] = field(default_factory=list)
    linked_issues: list[int] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format issue context for Claude prompt."""
        lines = [
            f"# Issue #{self.number}: {self.title}",
            "",
            "## Description",
            self.body or "(No description provided)",
            "",
        ]

        if self.labels:
            lines.extend(["## Labels", ", ".join(self.labels), ""])

        if self.comments:
            lines.append("## Discussion")
            for comment in self.comments:
                lines.extend([
                    f"### {comment.author} ({comment.created_at.strftime('%Y-%m-%d %H:%M')})",
                    comment.body,
                    "",
                ])

        if self.referenced_files:
            lines.extend([
                "## Referenced Files",
                "\n".join(f"- {f}" for f in self.referenced_files),
                "",
            ])

        if self.error_messages:
            lines.extend([
                "## Error Messages",
                "\n".join(f"```\n{e}\n```" for e in self.error_messages),
                "",
            ])

        return "\n".join(lines)


@dataclass
class ProcessingResult:
    """Result of processing a GitHub issue."""

    issue_number: int
    status: ProcessingStatus
    branch_name: Optional[str] = None
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    commits: list[str] = field(default_factory=list)
    blocking_question: Optional[str] = None
    error_message: Optional[str] = None
    ci_status: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Issue #{self.issue_number}: {self.status.value}"]

        if self.branch_name:
            lines.append(f"Branch: {self.branch_name}")

        if self.pr_url:
            lines.append(f"PR: {self.pr_url}")

        if self.commits:
            lines.append(f"Commits: {len(self.commits)}")

        if self.blocking_question:
            lines.append(f"Blocked: {self.blocking_question}")

        if self.error_message:
            lines.append(f"Error: {self.error_message}")

        return "\n".join(lines)


@dataclass
class BranchInfo:
    """Information about a git branch."""

    name: str
    issue_number: int
    created: bool = False
    pushed: bool = False
    commit_count: int = 0
    worktree_path: str | None = None  # Path to worktree if using isolated worktrees


@dataclass
class CIStatus:
    """Status of CI checks for a branch."""

    conclusion: Optional[str]  # success, failure, neutral, cancelled, timed_out, action_required, None (pending)
    status: str  # queued, in_progress, completed
    check_runs: list[dict] = field(default_factory=list)
    workflow_runs: list[dict] = field(default_factory=list)

    @property
    def is_pending(self) -> bool:
        return self.status in ("queued", "in_progress")

    @property
    def is_success(self) -> bool:
        return self.conclusion == "success"

    @property
    def is_failure(self) -> bool:
        return self.conclusion in ("failure", "timed_out", "action_required")

    def failure_summary(self) -> str:
        """Get summary of failures for fixing."""
        failures = []
        for run in self.check_runs:
            if run.get("conclusion") == "failure":
                failures.append(f"- {run.get('name', 'Unknown')}: {run.get('output', {}).get('summary', 'No details')}")
        return "\n".join(failures) if failures else "No failure details available"


@dataclass
class ClarificationOption:
    """An option for a clarification question."""

    label: str
    description: str = ""
    selected: bool = False


@dataclass
class ClarificationQuestion:
    """A structured clarification question for the issue author."""

    id: str  # e.g., "SCOPE", "DEPENDENCY"
    question: str
    options: list[ClarificationOption] = field(default_factory=list)
    allow_other: bool = True  # Allow free-text "Other" option
    answer: Optional[str] = None  # Filled in when answered

    def to_markdown(self) -> str:
        """Render question as GitHub-flavored markdown with checkboxes."""
        lines = [f"**[{self.id}]** {self.question}"]
        for opt in self.options:
            checkbox = "- [ ]" if not opt.selected else "- [x]"
            if opt.description:
                lines.append(f"{checkbox} **{opt.label}** - {opt.description}")
            else:
                lines.append(f"{checkbox} {opt.label}")
        if self.allow_other:
            lines.append("- [ ] Other: ___")
        return "\n".join(lines)

    @staticmethod
    def parse_answers(markdown: str, questions: list["ClarificationQuestion"]) -> dict[str, str]:
        """Parse checkbox selections from markdown comment."""
        import re
        answers = {}
        for q in questions:
            # Look for checked boxes after the question ID
            pattern = rf"\*\*\[{q.id}\]\*\*.*?(?=\*\*\[|$)"
            match = re.search(pattern, markdown, re.DOTALL)
            if match:
                section = match.group(0)
                # Find checked options
                checked = re.findall(r"- \[x\] \*?\*?([^*\n-]+)", section, re.IGNORECASE)
                if checked:
                    answers[q.id] = checked[0].strip()
                # Check for "Other:" with text
                other_match = re.search(r"- \[x\] Other:\s*(.+)", section)
                if other_match:
                    answers[q.id] = other_match.group(1).strip()
        return answers


@dataclass
class ClarificationRequest:
    """A request for clarification before implementation."""

    questions: list[ClarificationQuestion]
    intro: str = "Before I begin implementation, I need to clarify a few things:"
    ready_signal: str = "React with :+1: or comment 'ready' when answered."

    def to_markdown(self) -> str:
        """Render full clarification request as markdown."""
        lines = [
            "## Clarification Needed",
            "",
            self.intro,
            "",
        ]
        for i, q in enumerate(self.questions, 1):
            lines.append(f"### {i}. {q.question}")
            lines.append("")
            for opt in q.options:
                checkbox = "- [ ]"
                if opt.description:
                    lines.append(f"{checkbox} **{opt.label}** - {opt.description}")
                else:
                    lines.append(f"{checkbox} {opt.label}")
            if q.allow_other:
                lines.append("- [ ] Other: ___")
            lines.append("")

        lines.extend([
            "---",
            f"*{self.ready_signal}*",
        ])
        return "\n".join(lines)
