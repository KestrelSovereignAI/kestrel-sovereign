#!/usr/bin/env python3
"""
Intensive Training Cycle for Kestrel Agent Self-Improvement.

This script runs a continuous improvement loop:
1. Agent reflects on itself (layered reflection)
2. Creates GitHub issues for actionable insights
3. Monitors for issue closures (Claude Code implementations)
4. Re-runs reflection to verify improvements
5. Repeats until healthy or max iterations reached

Unlike the nightly sleep hook (long-term consolidation), this is for
intensive, rapid improvement sessions - like meditation/training.

Usage:
    python scripts/training_cycle.py agent_data/kestrel_prime.db

    # With options:
    python scripts/training_cycle.py agent_data/kestrel_prime.db --max-iterations=10
    python scripts/training_cycle.py agent_data/kestrel_prime.db --create-tickets
    python scripts/training_cycle.py agent_data/kestrel_prime.db --deep

Environment Variables:
    GITHUB_PAT or GITHUB_TOKEN - Required for ticket creation
    GITHUB_SELF_REPO - Target repo (default: Kestrel-Sovereign-AI/kestrel)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
# Load .env from project root explicitly (not CWD)
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_banner():
    """Print training cycle banner."""
    print(f"""
{Colors.CYAN}{'='*60}
  KESTREL INTENSIVE TRAINING CYCLE
  "Not sleep - meditation for rapid improvement"
{'='*60}{Colors.END}
""")


def print_iteration_header(iteration: int, max_iterations: int):
    """Print iteration header."""
    print(f"""
{Colors.BOLD}{'─'*60}
  ITERATION {iteration}/{max_iterations}
  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'─'*60}{Colors.END}
""")


async def run_reflection(agent, depth: str = "normal") -> dict:
    """Run layered reflection on the agent.

    Returns:
        dict with layer results and action items
    """
    logger.info(f"Running reflection (depth={depth})...")

    # Get the reflection feature
    reflection = agent.features.get("ReflectionFeature")
    if not reflection:
        logger.error("ReflectionFeature not loaded!")
        return {"error": "ReflectionFeature not available"}

    # Run reflection
    try:
        result = await reflection.reflect(scope="all", depth=depth)
        return result
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {"error": str(e)}


def format_reflection_summary(result: dict) -> str:
    """Format reflection result for display."""
    if result.get("error"):  # Only show error if non-empty
        return f"{Colors.RED}ERROR: {result['error']}{Colors.END}"

    lines = []

    # Layer summaries
    for layer_name in ["arms", "memory", "mind"]:
        layer = result.get(layer_name)
        if layer:
            # Derive status from has_critical/passed/failed
            if layer.get("has_critical"):
                status = "CRITICAL"
                status_color = Colors.RED
            elif layer.get("failed", 0) > 0:
                status = "DEGRADED"
                status_color = Colors.YELLOW
            else:
                status = "HEALTHY"
                status_color = Colors.GREEN
            lines.append(f"  {layer_name.upper():10} [{status_color}{status}{Colors.END}]")

            # Show checks
            for check in layer.get("checks", []):
                check_status = check.get("status", "?").lower()
                icon = "✓" if check_status == "pass" else "⚠" if check_status == "warn" else "✗" if check_status == "fail" else "○"
                msg = check.get("message", check.get("name", ""))[:50]
                lines.append(f"    {icon} {msg}")

    # Show if stopped early
    if result.get("stopped_at_layer"):
        lines.append(f"\n  {Colors.YELLOW}⚠ Stopped at layer: {result['stopped_at_layer']}{Colors.END}")

    # Action items
    actions = result.get("actions", [])
    if actions:
        lines.append(f"\n  {Colors.BOLD}ACTION ITEMS ({len(actions)} total):{Colors.END}")
        for action in actions[:5]:  # Show top 5
            priority = action.get("priority", "UNKNOWN")
            priority_color = Colors.RED if priority == "critical" else Colors.YELLOW if priority in ["high", "medium"] else Colors.BLUE
            title = action.get("title", action.get("description", ""))[:60]
            lines.append(f"    [{priority_color}{priority.upper()}{Colors.END}] {title}")
        if len(actions) > 5:
            lines.append(f"    ... and {len(actions) - 5} more")

    return "\n".join(lines)


async def get_health_score(result: dict) -> tuple[int, int, int]:
    """Extract health score from reflection result.

    Returns:
        (critical_count, warn_count, pass_count)
    """
    critical = 0
    warn = 0
    passed = 0

    for layer_name in ["arms", "memory", "mind"]:
        layer = result.get(layer_name)
        # Layer may be None if reflection stopped early
        if not layer:
            continue
        for check in layer.get("checks", []):
            status = check.get("status", "").lower()  # pass, fail, warn
            severity = (check.get("severity") or "").lower()  # critical, medium, etc.
            if status == "fail" and severity == "critical":
                critical += 1
            elif status in ["fail", "warn"]:
                warn += 1
            elif status == "pass":
                passed += 1

    return (critical, warn, passed)


async def create_tickets_from_actions(
    db_path: str,
    actions: list,
    github_token: str,
    repo: str,
    created_titles: set = None,  # Track across iterations to deduplicate
) -> list[str]:
    """Create GitHub issues from action items.

    Only creates tickets for MEDIUM+ priority issues that look like
    actual bugs or problems (not positive observations).

    Returns:
        List of created issue URLs
    """
    if not actions:
        return []

    if created_titles is None:
        created_titles = set()

    from kestrel_sovereign.features.github.client import GitHubClient

    client = GitHubClient(token=github_token)
    created_urls = []

    # Positive indicators - skip issues that are observations, not bugs
    positive_indicators = [
        "correctly", "appropriately", "as expected", "working",
        "proper", "good", "successfully", "pass", "healthy",
        "accessible", "intact", "responding", "enabled"
    ]

    # Vague summary patterns - skip these as they're not actionable
    import re
    vague_patterns = [
        r"^\d+ failures?, \d+ improvements?$",  # "2 failures, 1 improvements"
        r"^\d+ insights \(\d+ actionable\)$",   # "8 insights (3 actionable)"
        r"^investigate and fix$",               # Generic placeholder
        r"^no [\w\s]+ found$",                  # "No issues found"
    ]

    for action in actions:
        if not action.get("actionable", True):
            continue

        # ActionItem uses 'priority' field (Severity enum), not 'severity'
        priority = action.get("priority", "unknown")
        if hasattr(priority, "value"):  # Handle Severity enum
            priority = priority.value
        priority = str(priority).lower()

        # Create tickets for MEDIUM+ priority issues, or LOW if they have real content
        if priority not in ["critical", "high", "medium", "low"]:
            continue

        # Get description - prefer 'description' over 'message'
        message = action.get("description", action.get("message", "Unknown issue"))

        # For LOW priority, skip very short descriptions (likely not actionable)
        if priority == "low" and len(message) < 30:
            continue

        # Skip positive observations (not bugs)
        if any(ind in message.lower() for ind in positive_indicators):
            continue

        # Skip vague summary messages
        if any(re.match(pattern, message.lower().strip()) for pattern in vague_patterns):
            continue

        # Deduplicate based on first 60 chars of message
        title_key = message[:60].lower().strip()
        if title_key in created_titles:
            continue
        created_titles.add(title_key)

        suggested_fix = action.get("fix_description", action.get("suggested_fix", ""))
        file_path = action.get("file_path", "")

        # Build issue
        title = f"[Agent Training] {priority.upper()}: {message[:50]}"

        sections = []
        sections.append("## Training Cycle Issue")
        sections.append("")
        sections.append(f"**Priority:** {priority.upper()}")
        sections.append(f"**Detected:** {datetime.utcnow().isoformat()}Z")
        sections.append("")

        sections.append("## Description")
        sections.append("")
        sections.append(message)
        sections.append("")

        if suggested_fix:
            sections.append("## Suggested Fix")
            sections.append("")
            sections.append(suggested_fix)
            sections.append("")

        if file_path:
            sections.append("## Location")
            sections.append("")
            sections.append(f"`{file_path}`")
            sections.append("")

        sections.append("---")
        sections.append("*Created by Kestrel Agent intensive training cycle.*")

        body = "\n".join(sections)

        # Labels
        labels = ["agent-insight", "training-cycle"]
        if priority == "critical":
            labels.append("priority-critical")
        elif priority in ["high", "medium"]:
            labels.append("enhancement")

        try:
            result = await client.create_issue(
                repo=repo,
                title=title,
                body=body,
                labels=labels,
            )
            url = result.get("html_url")
            if url:
                created_urls.append(url)
                logger.info(f"Created ticket: {url}")
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")

    return created_urls


async def get_agent_did(db_path: str) -> str:
    """Get the agent's DID from the database.

    Args:
        db_path: Path to the database file

    Returns:
        The agent's DID string
    """
    from kestrel_sovereign.storage import AsyncStorage

    storage = AsyncStorage(db_path)
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        if not agent_nodes:
            raise ValueError("No agent found in the database. Run inception_service.py first.")
        return agent_nodes[0].node_id
    finally:
        await storage.close()


async def run_training_cycle(
    db_path: str,
    max_iterations: int = 5,
    depth: str = "normal",
    create_tickets: bool = False,
    wait_between: float = 5.0,
    stop_when_healthy: bool = True,
):
    """Run the intensive training cycle.

    Args:
        db_path: Path to agent database
        max_iterations: Maximum number of reflection cycles
        depth: Reflection depth ("quick", "normal", "deep")
        create_tickets: Whether to create GitHub issues
        wait_between: Seconds to wait between iterations
        stop_when_healthy: Stop early if no critical/warn issues
    """
    print_banner()

    # Validate paths
    if os.path.isdir(db_path):
        db_path = os.path.join(db_path, "kestrel_prime.db")

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Check GitHub token if creating tickets
    github_token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_SELF_REPO", "Kestrel-Sovereign-AI/kestrel")

    if create_tickets and not github_token:
        logger.error("GITHUB_PAT or GITHUB_TOKEN required for ticket creation")
        sys.exit(1)

    # Get DID from database
    logger.info(f"Loading agent DID from {db_path}...")
    agent_did = await get_agent_did(db_path)
    logger.info(f"Agent DID: {agent_did}")

    # Initialize agent
    logger.info(f"Initializing agent...")

    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    llm_service = LLMService()
    agent = KestrelAgent(did=agent_did, storage_path=db_path, llm_service=llm_service)
    await agent.initialize()

    if not agent.features.get("ReflectionFeature"):
        logger.error("ReflectionFeature not available - check feature loading")
        sys.exit(1)

    print(f"""
{Colors.CYAN}Configuration:{Colors.END}
  Database:     {db_path}
  Max Iterations: {max_iterations}
  Depth:        {depth}
  Create Tickets: {create_tickets}
  Target Repo:  {repo if create_tickets else 'N/A'}
  Stop When Healthy: {stop_when_healthy}
""")

    # Training loop
    all_tickets = []
    iteration_results = []
    created_titles = set()  # Track created issue titles across iterations for deduplication

    for iteration in range(1, max_iterations + 1):
        print_iteration_header(iteration, max_iterations)

        # Run reflection
        result = await run_reflection(agent, depth=depth)

        # Display summary
        print(format_reflection_summary(result))

        # Get health score
        critical, warn, passed = await get_health_score(result)
        iteration_results.append({
            "iteration": iteration,
            "critical": critical,
            "warn": warn,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"\n  {Colors.BOLD}Health Score:{Colors.END} {Colors.RED if critical else Colors.GREEN}{passed} pass, {warn} warn, {critical} critical{Colors.END}")

        # Check if healthy
        if stop_when_healthy and critical == 0 and warn == 0:
            print(f"\n{Colors.GREEN}Agent is HEALTHY - training complete!{Colors.END}")
            break

        # Create tickets if enabled
        actions = result.get("actions", [])
        if create_tickets and actions:
            print(f"\n  Analyzing {len(actions)} action items for ticket creation...")
            tickets = await create_tickets_from_actions(
                db_path, actions, github_token, repo, created_titles
            )
            all_tickets.extend(tickets)
            if tickets:
                print(f"  Created {len(tickets)} tickets (filtered from {len(actions)} actions)")
            else:
                print(f"  No new tickets created (all filtered as duplicates or positive observations)")

        # Wait before next iteration (unless last)
        if iteration < max_iterations:
            print(f"\n  Waiting {wait_between}s before next iteration...")
            await asyncio.sleep(wait_between)

    # Final summary
    print(f"""
{Colors.CYAN}{'='*60}
  TRAINING CYCLE COMPLETE
{'='*60}{Colors.END}

  Iterations:     {len(iteration_results)}
  Tickets Created: {len(all_tickets)}

  Health Trend:
""")

    for r in iteration_results:
        status_char = "●" if r["critical"] == 0 and r["warn"] == 0 else "○" if r["critical"] == 0 else "✗"
        print(f"    Iter {r['iteration']}: {status_char} ({r['passed']} pass, {r['warn']} warn, {r['critical']} critical)")

    if all_tickets:
        print(f"\n  Tickets Created:")
        for url in all_tickets:
            print(f"    - {url}")

    print(f"""
{Colors.CYAN}{'='*60}{Colors.END}
""")

    # Cleanup
    await agent.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Kestrel Intensive Training Cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (3 iterations, normal depth)
  python scripts/training_cycle.py agent_data/kestrel_prime.db

  # Deep training with ticket creation
  python scripts/training_cycle.py agent_data/kestrel_prime.db --deep --create-tickets

  # Extended training session
  python scripts/training_cycle.py agent_data/kestrel_prime.db --max-iterations=20 --wait=10
        """
    )

    parser.add_argument("db_path", help="Path to agent database")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Maximum iterations (default: 5)")
    parser.add_argument("--deep", action="store_true",
                        help="Use deep reflection analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick reflection analysis")
    parser.add_argument("--create-tickets", action="store_true",
                        help="Create GitHub issues for action items")
    parser.add_argument("--wait", type=float, default=5.0,
                        help="Seconds between iterations (default: 5)")
    parser.add_argument("--no-stop-healthy", action="store_true",
                        help="Don't stop when healthy, run all iterations")

    args = parser.parse_args()

    depth = "deep" if args.deep else "quick" if args.quick else "normal"

    asyncio.run(run_training_cycle(
        db_path=args.db_path,
        max_iterations=args.max_iterations,
        depth=depth,
        create_tickets=args.create_tickets,
        wait_between=args.wait,
        stop_when_healthy=not args.no_stop_healthy,
    ))


if __name__ == "__main__":
    main()
