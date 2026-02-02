"""Multi-repo orchestrator for coordinated GitHub ticket processing.

The orchestrator manages processing across multiple repositories with awareness
of dependencies between them. For example, a framework repo may be upstream of
an application repo, so changes to the framework may require corresponding updates.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .config import GitHubProcessorConfig
from .github_client import GitHubClient
from .models import IssueContext, ProcessingResult, ProcessingStatus
from .ticket_processor import TicketProcessor


class RepoRelationship(Enum):
    """Relationship between repositories."""
    UPSTREAM = "upstream"  # Changes here may affect downstream
    DOWNSTREAM = "downstream"  # Depends on upstream repos
    INDEPENDENT = "independent"  # No dependency relationship


@dataclass
class RepoConfig:
    """Configuration for a single repository."""
    repo: str  # owner/repo format
    relationship: RepoRelationship = RepoRelationship.INDEPENDENT
    depends_on: list[str] = field(default_factory=list)  # List of upstream repos
    downstream: list[str] = field(default_factory=list)  # List of downstream repos
    assignee: Optional[str] = None  # Override bot assignee for this repo
    reviewer: Optional[str] = None  # Override human reviewer for this repo


@dataclass
class OrchestrationResult:
    """Result of orchestrating across multiple repos."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    results_by_repo: dict[str, list[ProcessingResult]] = field(default_factory=dict)
    cross_repo_issues: list[dict] = field(default_factory=list)  # Issues created in downstream repos

    def summary(self) -> str:
        """Generate summary of orchestration results."""
        lines = ["=" * 60, "ORCHESTRATION SUMMARY", "=" * 60]

        total_processed = 0
        total_completed = 0
        total_blocked = 0
        total_failed = 0

        for repo, results in self.results_by_repo.items():
            lines.append(f"\n{repo}:")
            for result in results:
                total_processed += 1
                status_emoji = {
                    ProcessingStatus.COMPLETED: "✓",
                    ProcessingStatus.BLOCKED: "⏸",
                    ProcessingStatus.FAILED: "✗",
                }.get(result.status, "?")

                lines.append(f"  {status_emoji} #{result.issue_number}: {result.status.value}")
                if result.pr_url:
                    lines.append(f"    PR: {result.pr_url}")
                if result.blocking_question:
                    lines.append(f"    Blocked: {result.blocking_question[:50]}...")

                if result.status == ProcessingStatus.COMPLETED:
                    total_completed += 1
                elif result.status == ProcessingStatus.BLOCKED:
                    total_blocked += 1
                elif result.status == ProcessingStatus.FAILED:
                    total_failed += 1

        if self.cross_repo_issues:
            lines.append(f"\nCross-repo issues created: {len(self.cross_repo_issues)}")
            for issue in self.cross_repo_issues:
                lines.append(f"  - {issue['repo']}#{issue['number']}: {issue['title']}")

        lines.extend([
            "",
            f"Total: {total_processed} | Completed: {total_completed} | Blocked: {total_blocked} | Failed: {total_failed}",
        ])

        return "\n".join(lines)


class Orchestrator:
    """Orchestrates GitHub ticket processing across multiple repositories."""

    def __init__(
        self,
        repos: list[RepoConfig],
        github_token: str,
        anthropic_api_key: str,
        default_assignee: str = "claude-bot",
        default_reviewer: str = "",
        model: str = "claude-opus-4-5-20251101",
        dry_run: bool = False,
    ):
        self.repos = {r.repo: r for r in repos}
        self.github_token = github_token
        self.anthropic_api_key = anthropic_api_key
        self.default_assignee = default_assignee
        self.default_reviewer = default_reviewer
        self.model = model
        self.dry_run = dry_run

        # Build dependency graph
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build the dependency graph between repos."""
        # Determine processing order (upstream first)
        self.processing_order: list[str] = []
        visited = set()

        def visit(repo: str):
            if repo in visited:
                return
            visited.add(repo)

            config = self.repos.get(repo)
            if config:
                for dep in config.depends_on:
                    visit(dep)

            self.processing_order.append(repo)

        for repo in self.repos:
            visit(repo)

    def _create_config_for_repo(self, repo_config: RepoConfig) -> GitHubProcessorConfig:
        """Create a GitHubProcessorConfig for a specific repo."""
        return GitHubProcessorConfig(
            github_token=self.github_token,
            anthropic_api_key=self.anthropic_api_key,
            repo=repo_config.repo,
            bot_assignee=repo_config.assignee or self.default_assignee,
            human_reviewer=repo_config.reviewer or self.default_reviewer,
            model=self.model,
            dry_run=self.dry_run,
        )

    async def process_all(self) -> OrchestrationResult:
        """Process all assigned issues across all repos in dependency order."""
        result = OrchestrationResult(started_at=datetime.now())

        print(f"Processing {len(self.repos)} repositories in order: {self.processing_order}")

        for repo in self.processing_order:
            repo_config = self.repos[repo]
            print(f"\n{'='*60}")
            print(f"Processing: {repo}")
            print(f"{'='*60}")

            config = self._create_config_for_repo(repo_config)
            processor = TicketProcessor(config)

            repo_results = await processor.process_all_assigned()
            result.results_by_repo[repo] = repo_results

            # Check if any completed changes might affect downstream repos
            completed_results = [r for r in repo_results if r.status == ProcessingStatus.COMPLETED]
            if completed_results and repo_config.downstream:
                await self._check_downstream_impact(
                    repo, completed_results, repo_config.downstream, result
                )

        result.completed_at = datetime.now()
        return result

    async def process_repo(self, repo: str) -> OrchestrationResult:
        """Process a single repository."""
        result = OrchestrationResult(started_at=datetime.now())

        if repo not in self.repos:
            print(f"Repository {repo} not configured")
            result.completed_at = datetime.now()
            return result

        repo_config = self.repos[repo]
        config = self._create_config_for_repo(repo_config)
        processor = TicketProcessor(config)

        repo_results = await processor.process_all_assigned()
        result.results_by_repo[repo] = repo_results

        # Check downstream impact
        completed_results = [r for r in repo_results if r.status == ProcessingStatus.COMPLETED]
        if completed_results and repo_config.downstream:
            await self._check_downstream_impact(
                repo, completed_results, repo_config.downstream, result
            )

        result.completed_at = datetime.now()
        return result

    async def process_issue(self, repo: str, issue_number: int) -> OrchestrationResult:
        """Process a specific issue in a repository."""
        result = OrchestrationResult(started_at=datetime.now())

        if repo not in self.repos:
            print(f"Repository {repo} not configured")
            result.completed_at = datetime.now()
            return result

        repo_config = self.repos[repo]
        config = self._create_config_for_repo(repo_config)
        config.issue_number = issue_number

        processor = TicketProcessor(config)
        issue_result = await processor.process_single(issue_number)
        result.results_by_repo[repo] = [issue_result]

        # Check downstream impact
        if issue_result.status == ProcessingStatus.COMPLETED and repo_config.downstream:
            await self._check_downstream_impact(
                repo, [issue_result], repo_config.downstream, result
            )

        result.completed_at = datetime.now()
        return result

    async def _check_downstream_impact(
        self,
        upstream_repo: str,
        completed_results: list[ProcessingResult],
        downstream_repos: list[str],
        orchestration_result: OrchestrationResult,
    ) -> None:
        """Check if completed changes might require updates to downstream repos.

        This creates issues in downstream repos when upstream changes might affect them.
        """
        if self.dry_run:
            print(f"[DRY RUN] Would check downstream impact on: {downstream_repos}")
            return

        for downstream_repo in downstream_repos:
            if downstream_repo not in self.repos:
                continue

            downstream_config = self.repos[downstream_repo]
            config = self._create_config_for_repo(downstream_config)
            github = GitHubClient(config)

            for result in completed_results:
                if result.pr_url:
                    # Create an issue in the downstream repo to review the upstream change
                    issue_body = f"""## Upstream Change Notification

An upstream change in `{upstream_repo}` may require updates in this repository.

**Upstream PR:** {result.pr_url}
**Upstream Issue:** #{result.issue_number}

### Action Required

Please review the upstream change and determine if any updates are needed here:
1. Check if APIs or interfaces used from `{upstream_repo}` have changed
2. Update any affected code
3. Run tests to verify compatibility

---
*This issue was automatically created by the GitHub Ticket Orchestrator.*
"""

                    title = f"Review upstream change from {upstream_repo}#{result.issue_number}"

                    # Create the issue (in a real implementation, we'd use the GitHub API)
                    print(f"Creating downstream issue in {downstream_repo}: {title}")

                    # Note: For now we just log this. Full implementation would:
                    # issue = github.repo.create_issue(title=title, body=issue_body)
                    # orchestration_result.cross_repo_issues.append({
                    #     "repo": downstream_repo,
                    #     "number": issue.number,
                    #     "title": title,
                    #     "upstream_pr": result.pr_url,
                    # })


def create_multi_repo_orchestrator(
    github_token: str,
    anthropic_api_key: str,
    upstream_repo: str = "KestrelSovereignAI/kestrel-sovereign",
    downstream_repo: str | None = None,
    assignee: str = "claude-bot",
    reviewer: str = "",
    dry_run: bool = False,
) -> Orchestrator:
    """Create an orchestrator for multi-repo processing.

    When downstream_repo is provided, upstream changes may require corresponding updates.
    """
    repos = [
        RepoConfig(
            repo=upstream_repo,
            relationship=RepoRelationship.UPSTREAM if downstream_repo else RepoRelationship.INDEPENDENT,
            downstream=[downstream_repo] if downstream_repo else [],
        ),
    ]

    if downstream_repo:
        repos.append(
            RepoConfig(
                repo=downstream_repo,
                relationship=RepoRelationship.DOWNSTREAM,
                depends_on=[upstream_repo],
            )
        )

    return Orchestrator(
        repos=repos,
        github_token=github_token,
        anthropic_api_key=anthropic_api_key,
        default_assignee=assignee,
        default_reviewer=reviewer,
        dry_run=dry_run,
    )
