"""Configuration for GitHub Ticket Processor."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GitHubProcessorConfig:
    """Configuration for the GitHub ticket processor."""

    # GitHub settings
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    repo: str = ""  # Format: "owner/repo"
    bot_assignee: str = field(
        default_factory=lambda: os.environ.get("GITHUB_BOT_ASSIGNEE", "claude-bot")
    )
    human_reviewer: str = field(
        default_factory=lambda: os.environ.get("GITHUB_HUMAN_REVIEWER", "")
    )

    # Claude Agent SDK settings
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model: str = "claude-opus-4-5-20251101"
    max_turns: int = 50  # Prevent infinite loops

    # Processing settings
    max_ci_retries: int = 3  # Max attempts to fix CI failures
    ci_poll_interval: int = 30  # Seconds between CI status checks
    ci_timeout: int = 600  # Max seconds to wait for CI

    # Labels for agent coordination (no bot account needed)
    label_analyzing: str = "agent-analyzing"      # Agent reviewing, may ask questions
    label_clarifying: str = "agent-clarifying"    # Agent waiting for answers
    label_in_progress: str = "agent-claimed"      # Agent is actively working
    label_blocked: str = "agent-blocked"          # Agent hit a blocker, needs human
    label_failed: str = "agent-blocked"           # Reuse blocked for failures too
    label_completed: str = "agent-complete"       # Agent finished, PR ready

    # Behavior
    dry_run: bool = False
    post_plan_comment: bool = True  # Post implementation plan as comment
    create_draft_pr: bool = False  # Create draft PR instead of regular PR
    skip_clarification: bool = False  # Skip clarification phase (for agent-ready issues)
    clarification_timeout: int = 86400  # Seconds to wait for clarification (24h)

    # Specific issue to process (None = process all assigned)
    issue_number: Optional[int] = None

    # Worktree settings for isolation
    use_worktree: bool = False  # Create isolated worktree instead of checkout
    worktree_base_path: str = "../"  # Base path for worktree directories
    worktree_path: Optional[str] = None  # Set at runtime when worktree is created

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.github_token:
            errors.append("GITHUB_TOKEN environment variable is required")

        if not self.repo:
            errors.append("Repository (--repo) is required")

        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY environment variable is required")

        if "/" not in self.repo and self.repo:
            errors.append("Repository must be in format 'owner/repo'")

        return errors

    @property
    def repo_owner(self) -> str:
        """Get repository owner from repo string."""
        if "/" in self.repo:
            return self.repo.split("/")[0]
        return ""

    @property
    def repo_name(self) -> str:
        """Get repository name from repo string."""
        if "/" in self.repo:
            return self.repo.split("/")[1]
        return ""
