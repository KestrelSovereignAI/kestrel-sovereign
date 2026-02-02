"""GitHub API client for ticket processing."""

import re
import subprocess
from datetime import datetime
from typing import Optional

from github import Auth, Github
from github.Issue import Issue
from github.Repository import Repository

from .config import GitHubProcessorConfig
from .models import BranchInfo, CIStatus, IssueComment, IssueContext


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(self, config: GitHubProcessorConfig):
        self.config = config
        self._github = Github(auth=Auth.Token(config.github_token))
        self._repo: Optional[Repository] = None

    @property
    def repo(self) -> Repository:
        """Get the GitHub repository."""
        if self._repo is None:
            self._repo = self._github.get_repo(self.config.repo)
        return self._repo

    def get_assigned_issues(self) -> list[Issue]:
        """Get all open issues assigned to the bot."""
        issues = self.repo.get_issues(
            state="open",
            assignee=self.config.bot_assignee,
        )
        return list(issues)

    def get_claimable_issues(self, require_label: str = "enhancement") -> list[Issue]:
        """Get open issues that can be claimed by an agent.

        Returns issues that:
        - Have the required label (e.g., 'enhancement')
        - Do NOT have 'agent-claimed', 'agent-blocked', or 'agent-complete' labels
        """
        issues = self.repo.get_issues(state="open", labels=[require_label])

        # Filter out already-claimed issues
        agent_labels = {
            self.config.label_in_progress,
            self.config.label_blocked,
            self.config.label_completed,
        }

        claimable = []
        for issue in issues:
            issue_labels = {label.name for label in issue.labels}
            if not issue_labels.intersection(agent_labels):
                claimable.append(issue)

        return claimable

    def get_issues_with_label(self, label: str) -> list[Issue]:
        """Get all open issues with a specific label."""
        issues = self.repo.get_issues(state="open", labels=[label])
        return list(issues)

    def is_claimed(self, issue_number: int) -> bool:
        """Check if an issue is already claimed by an agent."""
        issue = self.get_issue(issue_number)
        issue_labels = {label.name for label in issue.labels}
        return self.config.label_in_progress in issue_labels

    def get_issue(self, issue_number: int) -> Issue:
        """Get a specific issue by number."""
        return self.repo.get_issue(issue_number)

    def build_issue_context(self, issue: Issue) -> IssueContext:
        """Build full context from an issue for processing."""
        comments = []
        for comment in issue.get_comments():
            comments.append(IssueComment(
                id=comment.id,
                author=comment.user.login if comment.user else "unknown",
                body=comment.body or "",
                created_at=comment.created_at,
                is_bot=comment.user.type == "Bot" if comment.user else False,
            ))

        context = IssueContext(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            author=issue.user.login if issue.user else "unknown",
            labels=[label.name for label in issue.labels],
            assignees=[a.login for a in issue.assignees],
            comments=comments,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            url=issue.html_url,
        )

        # Extract referenced files from issue body and comments
        context.referenced_files = self._extract_file_references(context)
        context.error_messages = self._extract_error_messages(context)
        context.linked_prs = self._extract_linked_prs(context)
        context.linked_issues = self._extract_linked_issues(context)

        return context

    def _extract_file_references(self, context: IssueContext) -> list[str]:
        """Extract file paths mentioned in issue."""
        all_text = context.body + "\n" + "\n".join(c.body for c in context.comments)

        # Match common file path patterns
        patterns = [
            r'`([a-zA-Z0-9_\-./]+\.[a-zA-Z]+)`',  # `path/to/file.ext`
            r'(?:^|\s)([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-./]+\.[a-zA-Z]+)',  # path/to/file.ext
            r'(?:in|at|file|see)\s+[`"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z]+)[`"]?',  # "in file.py"
        ]

        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, all_text, re.MULTILINE)
            files.update(matches)

        return sorted(files)

    def _extract_error_messages(self, context: IssueContext) -> list[str]:
        """Extract error messages/stack traces from issue."""
        all_text = context.body + "\n" + "\n".join(c.body for c in context.comments)

        # Extract code blocks that look like errors
        code_blocks = re.findall(r'```(?:[\w]*\n)?(.*?)```', all_text, re.DOTALL)
        errors = []

        error_indicators = ['error', 'exception', 'traceback', 'failed', 'fatal']
        for block in code_blocks:
            if any(indicator in block.lower() for indicator in error_indicators):
                errors.append(block.strip())

        return errors

    def _extract_linked_prs(self, context: IssueContext) -> list[int]:
        """Extract linked PR numbers from issue."""
        all_text = context.body + "\n" + "\n".join(c.body for c in context.comments)
        # Match #123 or PR #123 or pull/123
        matches = re.findall(r'(?:PR\s*#?|pull/|#)(\d+)', all_text, re.IGNORECASE)
        return sorted(set(int(m) for m in matches if int(m) != context.number))

    def _extract_linked_issues(self, context: IssueContext) -> list[int]:
        """Extract linked issue numbers."""
        all_text = context.body + "\n" + "\n".join(c.body for c in context.comments)
        matches = re.findall(r'(?:issue\s*#?|fixes\s*#?|closes\s*#?|relates?\s*to\s*#?)(\d+)', all_text, re.IGNORECASE)
        return sorted(set(int(m) for m in matches if int(m) != context.number))

    def add_comment(self, issue_number: int, body: str) -> None:
        """Add a comment to an issue."""
        if self.config.dry_run:
            print(f"[DRY RUN] Would add comment to #{issue_number}:\n{body[:200]}...")
            return

        issue = self.get_issue(issue_number)
        issue.create_comment(body)

    def add_label(self, issue_number: int, label: str) -> None:
        """Add a label to an issue."""
        if self.config.dry_run:
            print(f"[DRY RUN] Would add label '{label}' to #{issue_number}")
            return

        issue = self.get_issue(issue_number)
        issue.add_to_labels(label)

    def remove_label(self, issue_number: int, label: str) -> None:
        """Remove a label from an issue."""
        if self.config.dry_run:
            print(f"[DRY RUN] Would remove label '{label}' from #{issue_number}")
            return

        issue = self.get_issue(issue_number)
        try:
            issue.remove_from_labels(label)
        except Exception:
            pass  # Label might not exist

    def set_assignees(self, issue_number: int, assignees: list[str]) -> None:
        """Set assignees on an issue (replaces existing)."""
        if self.config.dry_run:
            print(f"[DRY RUN] Would set assignees on #{issue_number}: {assignees}")
            return

        issue = self.get_issue(issue_number)
        # Remove existing assignees
        for assignee in issue.assignees:
            issue.remove_from_assignees(assignee)
        # Add new assignees
        for assignee in assignees:
            issue.add_to_assignees(assignee)

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        draft: bool = False,
    ) -> tuple[int, str]:
        """Create a pull request. Returns (pr_number, pr_url)."""
        if self.config.dry_run:
            print(f"[DRY RUN] Would create PR: {title}")
            return (0, "https://github.com/dry-run/pr")

        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
            draft=draft,
        )
        return (pr.number, pr.html_url)

    def get_ci_status(self, branch: str) -> CIStatus:
        """Get CI status for a branch."""
        # Get the latest commit on the branch
        try:
            branch_ref = self.repo.get_branch(branch)
            sha = branch_ref.commit.sha
        except Exception:
            return CIStatus(conclusion=None, status="unknown")

        # Get check runs for the commit
        check_runs = list(self.repo.get_commit(sha).get_check_runs())

        # Determine overall status
        if not check_runs:
            # No CI configured - treat as success (don't wait forever)
            return CIStatus(conclusion="success", status="completed", check_runs=[])

        conclusions = [run.conclusion for run in check_runs if run.conclusion]
        statuses = [run.status for run in check_runs]

        if "in_progress" in statuses or "queued" in statuses:
            status = "in_progress"
            conclusion = None
        elif all(c == "success" for c in conclusions):
            status = "completed"
            conclusion = "success"
        elif any(c == "failure" for c in conclusions):
            status = "completed"
            conclusion = "failure"
        else:
            status = "completed"
            conclusion = conclusions[0] if conclusions else None

        return CIStatus(
            conclusion=conclusion,
            status=status,
            check_runs=[{
                "name": run.name,
                "conclusion": run.conclusion,
                "status": run.status,
                "output": {
                    "title": run.output.title if run.output else None,
                    "summary": run.output.summary if run.output else None,
                },
            } for run in check_runs],
        )


class GitOperations:
    """Git operations for branch and commit management."""

    def __init__(self, config: GitHubProcessorConfig):
        self.config = config
        self.worktree_path: str | None = None  # Set when worktree is created

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command, optionally in the worktree directory."""
        cmd = ["git"] + list(args)
        cwd = self.worktree_path if self.worktree_path else None
        if self.config.dry_run:
            cwd_msg = f" (in {cwd})" if cwd else ""
            print(f"[DRY RUN] Would run: {' '.join(cmd)}{cwd_msg}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(cmd, capture_output=True, text=True, check=check, cwd=cwd)

    def create_worktree(self, issue_number: int, title: str) -> BranchInfo:
        """Create a new git worktree for an issue (isolated from main repo).

        This creates a separate working directory with its own branch,
        allowing parallel work on multiple issues without conflicts.

        Args:
            issue_number: The GitHub issue number
            title: Issue title (used for branch naming)

        Returns:
            BranchInfo with branch name and worktree path
        """
        import os

        # Slugify the title
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower())[:30].strip('-')
        branch_name = f"issue-{issue_number}-{slug}"

        # Determine worktree path
        base_path = self.config.worktree_base_path
        repo_name = self.config.repo_name or "repo"
        worktree_dir = f"{repo_name}-issue-{issue_number}"
        worktree_path = os.path.abspath(os.path.join(base_path, worktree_dir))

        # Fetch latest from origin (run in main repo, not worktree)
        self._run_git("fetch", "origin")

        # Create worktree with new branch from main
        result = self._run_git(
            "worktree", "add", worktree_path, "-b", branch_name, "origin/main",
            check=False
        )

        if result.returncode != 0:
            # Check if branch already exists
            if "already exists" in result.stderr:
                # Try to add worktree with existing branch
                result = self._run_git(
                    "worktree", "add", worktree_path, branch_name,
                    check=False
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to create worktree: {result.stderr}")
            else:
                raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        # Store worktree path for subsequent operations
        self.worktree_path = worktree_path
        self.config.worktree_path = worktree_path

        print(f"Created worktree at: {worktree_path}")

        return BranchInfo(
            name=branch_name,
            issue_number=issue_number,
            created=True,
            worktree_path=worktree_path,
        )

    def create_branch(self, issue_number: int, title: str) -> BranchInfo:
        """Create a new branch for an issue.

        If use_worktree is enabled in config, creates an isolated worktree.
        Otherwise, creates a branch with checkout in the current directory.
        """
        if self.config.use_worktree:
            return self.create_worktree(issue_number, title)

        # Slugify the title
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower())[:30].strip('-')
        branch_name = f"issue-{issue_number}-{slug}"

        # Fetch latest from origin
        self._run_git("fetch", "origin")

        # Create branch from main
        self._run_git("checkout", "-b", branch_name, "origin/main")

        return BranchInfo(
            name=branch_name,
            issue_number=issue_number,
            created=True,
        )

    def checkout_branch(self, branch_name: str) -> None:
        """Checkout an existing branch."""
        self._run_git("checkout", branch_name)

    def current_branch(self) -> str:
        """Get current branch name."""
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def push_branch(self, branch_name: str) -> None:
        """Push branch to origin."""
        self._run_git("push", "-u", "origin", branch_name)

    def commit(self, message: str, issue_number: int) -> str:
        """Create a commit with issue reference. Returns commit SHA."""
        # Ensure issue reference is in message
        if f"#{issue_number}" not in message:
            message = f"{message} (#{issue_number})"

        self._run_git("add", "-A")
        result = self._run_git("commit", "-m", message, check=False)

        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                return ""
            raise RuntimeError(f"Commit failed: {result.stderr}")

        # Get commit SHA
        sha_result = self._run_git("rev-parse", "HEAD")
        return sha_result.stdout.strip()

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        result = self._run_git("status", "--porcelain")
        return bool(result.stdout.strip())

    def get_commit_count(self, base: str = "origin/main") -> int:
        """Get number of commits ahead of base."""
        result = self._run_git("rev-list", "--count", f"{base}..HEAD")
        return int(result.stdout.strip())
