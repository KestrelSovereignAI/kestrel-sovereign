#!/usr/bin/env python3
"""CLI entry point for GitHub Ticket Processor and Orchestrator."""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # Load .env before reading config

from .config import GitHubProcessorConfig
from .models import ProcessingStatus
from .orchestrator import Orchestrator, RepoConfig, RepoRelationship, create_multi_repo_orchestrator
from .ticket_processor import TicketProcessor


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="GitHub Ticket Processor - Autonomous issue processing with Claude Agent SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Claim and process issues (label-based, no bot account needed)
    claim_parser = subparsers.add_parser(
        "claim",
        help="Claim and process issues using labels (no bot account needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all claimable 'enhancement' issues
  kestrel-github claim --repo owner/repo

  # Process issues with a specific label
  kestrel-github claim --repo owner/repo --label bug

  # Process a specific issue by number
  kestrel-github claim --repo owner/repo --issue 42

  # Use isolated worktree (recommended for parallel processing)
  kestrel-github claim --repo owner/repo --issue 42 --worktree

  # Dry run
  kestrel-github claim --repo owner/repo --dry-run

Label-based coordination:
  - agent-claimed: Issue is being worked on
  - agent-blocked: Agent needs human input
  - agent-complete: Agent finished, PR ready

Worktree mode (--worktree):
  Creates isolated git worktrees for each issue, allowing multiple
  agents to work in parallel without conflicts. Worktrees are created
  in the parent directory by default (use --worktree-base to change).
""",
    )
    _add_common_args(claim_parser)
    claim_parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in 'owner/repo' format",
    )
    claim_parser.add_argument(
        "--issue",
        type=int,
        help="Specific issue number to claim and process",
    )
    claim_parser.add_argument(
        "--label",
        default="enhancement",
        help="Label to filter claimable issues (default: enhancement)",
    )

    # Single repo processing (legacy, assignee-based)
    process_parser = subparsers.add_parser(
        "process",
        help="Process issues assigned to bot (legacy, use 'claim' instead)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all issues assigned to claude-bot
  kestrel-github process --repo owner/repo --assignee claude-bot

  # Process a specific issue
  kestrel-github process --repo owner/repo --issue 42

  # Dry run (show what would be done)
  kestrel-github process --repo owner/repo --dry-run

NOTE: Consider using 'claim' command instead - it uses labels for
coordination and doesn't require a separate bot account.
""",
    )
    _add_common_args(process_parser)
    process_parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in 'owner/repo' format",
    )
    process_parser.add_argument(
        "--issue",
        type=int,
        help="Specific issue number to process (default: all assigned)",
    )

    # Orchestrator for multi-repo processing
    orch_parser = subparsers.add_parser(
        "orchestrate",
        help="Orchestrate processing across multiple repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process issues in upstream repo
  kestrel-github orchestrate --upstream owner/repo

  # Process upstream and downstream repos (upstream first)
  kestrel-github orchestrate --upstream owner/upstream --downstream owner/downstream

  # Process only upstream
  kestrel-github orchestrate --upstream owner/repo --repo upstream

  # Dry run
  kestrel-github orchestrate --upstream owner/repo --dry-run
""",
    )
    _add_common_args(orch_parser)
    orch_parser.add_argument(
        "--upstream",
        dest="upstream_repo",
        default="KestrelSovereignAI/kestrel-sovereign",
        help="Upstream repository (default: KestrelSovereignAI/kestrel-sovereign)",
    )
    orch_parser.add_argument(
        "--downstream",
        dest="downstream_repo",
        default=None,
        help="Downstream repository (optional)",
    )
    orch_parser.add_argument(
        "--repo",
        choices=["upstream", "downstream", "all"],
        default="all",
        help="Which repo(s) to process (default: all)",
    )
    orch_parser.add_argument(
        "--issue",
        type=int,
        help="Process a specific issue (requires --repo to be upstream or downstream)",
    )

    # Multi-repo orchestrator (custom repos)
    multi_parser = subparsers.add_parser(
        "multi",
        help="Orchestrate processing across multiple custom repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple independent repos
  kestrel-github multi --repos owner/repo1 owner/repo2

  # Define dependencies (repo2 depends on repo1)
  kestrel-github multi --repos owner/repo1 owner/repo2 --upstream owner/repo1
""",
    )
    _add_common_args(multi_parser)
    multi_parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="List of repositories to process (owner/repo format)",
    )
    multi_parser.add_argument(
        "--upstream",
        nargs="*",
        default=[],
        help="Repositories that are upstream (processed first)",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--assignee",
        default=os.environ.get("GITHUB_BOT_ASSIGNEE", "claude-bot"),
        help="Bot assignee to look for (default: claude-bot)",
    )
    parser.add_argument(
        "--reviewer",
        default=os.environ.get("GITHUB_HUMAN_REVIEWER", ""),
        help="Human reviewer for blocked issues",
    )
    parser.add_argument(
        "--model",
        choices=["opus", "sonnet", "haiku"],
        default="opus",
        help="Claude model to use (default: opus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum agent turns per issue (default: 50)",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Create isolated git worktree for each issue (recommended for parallel processing)",
    )
    parser.add_argument(
        "--worktree-base",
        default="../",
        help="Base path for worktree directories (default: ../)",
    )


def get_model_id(model_name: str) -> str:
    """Convert model name to model ID."""
    return {
        "opus": "claude-opus-4-5-20251101",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-3-20240307",
    }[model_name]


def validate_env() -> tuple[str, str]:
    """Validate required environment variables. Returns (github_token, anthropic_key)."""
    github_token = os.environ.get("GITHUB_TOKEN", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    errors = []
    if not github_token:
        errors.append("GITHUB_TOKEN environment variable is required")
    if not anthropic_key:
        errors.append("ANTHROPIC_API_KEY environment variable is required")

    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    return github_token, anthropic_key


async def cmd_claim(args: argparse.Namespace) -> int:
    """Handle the 'claim' command for label-based processing (no bot account needed)."""
    github_token, anthropic_key = validate_env()

    config = GitHubProcessorConfig(
        github_token=github_token,
        anthropic_api_key=anthropic_key,
        repo=args.repo,
        bot_assignee=args.assignee,  # Still used for some operations
        human_reviewer=args.reviewer,
        model=get_model_id(args.model),
        max_turns=args.max_turns,
        dry_run=args.dry_run,
        issue_number=args.issue,
        use_worktree=args.worktree,
        worktree_base_path=args.worktree_base,
    )

    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    processor = TicketProcessor(config)
    print(f"Agent session: {processor.session_id}")
    if config.use_worktree:
        print(f"Using worktrees (base: {config.worktree_base_path})")

    if config.issue_number:
        # Check if already claimed
        if processor.github.is_claimed(config.issue_number):
            print(f"Issue #{config.issue_number} is already claimed by another agent")
            return 1
        print(f"Claiming and processing issue #{config.issue_number} in {args.repo}...")
        result = await processor.process_single(config.issue_number)
        results = [result]
    else:
        print(f"Finding claimable issues in {args.repo} with label '{args.label}'...")
        results = await processor.process_claimable(args.label)

    # Print summary
    _print_results_summary(results)

    failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
    return 1 if failed > 0 else 0


async def cmd_process(args: argparse.Namespace) -> int:
    """Handle the 'process' command for single repo (legacy, assignee-based)."""
    github_token, anthropic_key = validate_env()

    config = GitHubProcessorConfig(
        github_token=github_token,
        anthropic_api_key=anthropic_key,
        repo=args.repo,
        bot_assignee=args.assignee,
        human_reviewer=args.reviewer,
        model=get_model_id(args.model),
        max_turns=args.max_turns,
        dry_run=args.dry_run,
        issue_number=args.issue,
        use_worktree=args.worktree,
        worktree_base_path=args.worktree_base,
    )

    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    processor = TicketProcessor(config)
    if config.use_worktree:
        print(f"Using worktrees (base: {config.worktree_base_path})")

    if config.issue_number:
        print(f"Processing issue #{config.issue_number} in {args.repo}...")
        result = await processor.process_single(config.issue_number)
        results = [result]
    else:
        print(f"Processing all assigned issues in {args.repo}...")
        results = await processor.process_all_assigned()

    # Print summary
    _print_results_summary(results)

    failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
    return 1 if failed > 0 else 0


async def cmd_orchestrate(args: argparse.Namespace) -> int:
    """Handle the 'orchestrate' command for multi-repo processing."""
    github_token, anthropic_key = validate_env()

    orchestrator = create_multi_repo_orchestrator(
        github_token=github_token,
        anthropic_api_key=anthropic_key,
        upstream_repo=args.upstream_repo,
        downstream_repo=args.downstream_repo,
        assignee=args.assignee,
        reviewer=args.reviewer,
        dry_run=args.dry_run,
    )

    if args.issue:
        if args.repo == "all":
            print("Error: --issue requires --repo to be 'upstream' or 'downstream'", file=sys.stderr)
            return 1

        repo = args.upstream_repo if args.repo == "upstream" else args.downstream_repo
        if not repo:
            print("Error: downstream repo not specified", file=sys.stderr)
            return 1
        print(f"Processing issue #{args.issue} in {repo}...")
        result = await orchestrator.process_issue(repo, args.issue)
    elif args.repo == "all":
        print("Orchestrating across repositories...")
        result = await orchestrator.process_all()
    else:
        repo = args.upstream_repo if args.repo == "upstream" else args.downstream_repo
        if not repo:
            print("Error: downstream repo not specified", file=sys.stderr)
            return 1
        print(f"Processing all assigned issues in {repo}...")
        result = await orchestrator.process_repo(repo)

    print(result.summary())

    # Check for failures
    for results in result.results_by_repo.values():
        if any(r.status == ProcessingStatus.FAILED for r in results):
            return 1

    return 0


async def cmd_multi(args: argparse.Namespace) -> int:
    """Handle the 'multi' command for custom multi-repo orchestration."""
    github_token, anthropic_key = validate_env()

    # Build repo configs
    upstream_set = set(args.upstream)
    repos = []
    for repo in args.repos:
        if repo in upstream_set:
            repos.append(RepoConfig(
                repo=repo,
                relationship=RepoRelationship.UPSTREAM,
                downstream=[r for r in args.repos if r not in upstream_set],
            ))
        else:
            repos.append(RepoConfig(
                repo=repo,
                relationship=RepoRelationship.DOWNSTREAM if upstream_set else RepoRelationship.INDEPENDENT,
                depends_on=list(upstream_set),
            ))

    orchestrator = Orchestrator(
        repos=repos,
        github_token=github_token,
        anthropic_api_key=anthropic_key,
        default_assignee=args.assignee,
        default_reviewer=args.reviewer,
        model=get_model_id(args.model),
        dry_run=args.dry_run,
    )

    print(f"Orchestrating across {len(repos)} repositories...")
    result = await orchestrator.process_all()
    print(result.summary())

    # Check for failures
    for results in result.results_by_repo.values():
        if any(r.status == ProcessingStatus.FAILED for r in results):
            return 1

    return 0


def _print_results_summary(results: list) -> None:
    """Print summary of processing results."""
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)

    completed = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
    blocked = sum(1 for r in results if r.status == ProcessingStatus.BLOCKED)
    failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)

    print(f"Total: {len(results)}")
    print(f"Completed: {completed}")
    print(f"Blocked: {blocked}")
    print(f"Failed: {failed}")

    for result in results:
        print(f"\n#{result.issue_number}: {result.status.value}")
        if result.pr_url:
            print(f"  PR: {result.pr_url}")
        if result.blocking_question:
            print(f"  Blocked: {result.blocking_question}")
        if result.error_message:
            print(f"  Error: {result.error_message}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made\n")

    if args.command == "claim":
        exit_code = asyncio.run(cmd_claim(args))
    elif args.command == "process":
        exit_code = asyncio.run(cmd_process(args))
    elif args.command == "orchestrate":
        exit_code = asyncio.run(cmd_orchestrate(args))
    elif args.command == "multi":
        exit_code = asyncio.run(cmd_multi(args))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
