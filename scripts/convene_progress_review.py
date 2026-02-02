#!/usr/bin/env python3
"""
Convene Constitutional Council for Project Progress Review

This script runs a formal council session with the configured foundation models
to review overall project progress and provide consensus feedback.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from kestrel_sovereign.features.council.models import Evidence, CouncilConfig, CouncilMember, ConsensusRule
from kestrel_sovereign.features.council.deliberation import convene_council
from kestrel_sovereign.features.council.storage import get_storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (January 2026)
MODEL_PRICING = {
    ("anthropic", "claude-opus-4-5-20251101"): {"input": 15.00, "output": 75.00},
    ("openai", "gpt-5.2"): {"input": 5.00, "output": 15.00},
    ("vertex_ai", "gemini-3-pro-preview"): {"input": 1.25, "output": 5.00},
    ("anthropic", "default"): {"input": 15.00, "output": 75.00},
    ("openai", "default"): {"input": 5.00, "output": 15.00},
    ("google", "default"): {"input": 1.25, "output": 5.00},
    ("vertex_ai", "default"): {"input": 1.25, "output": 5.00},
    ("ollama", "default"): {"input": 0.00, "output": 0.00},
}


def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for token usage."""
    key = (provider, model)
    if key not in MODEL_PRICING:
        key = (provider, "default")
    if key not in MODEL_PRICING:
        key = ("openai", "default")

    prices = MODEL_PRICING[key]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def print_token_usage(session) -> float:
    """Print token usage summary and return total cost."""
    print()
    print("=" * 70)
    print("TOKEN USAGE & COST SUMMARY")
    print("=" * 70)
    print()

    by_member = session.tokens_by_member()
    total_cost = 0.0

    print(f"{'Member':<12} {'Provider':<12} {'Input':>10} {'Output':>10} {'Est. Cost':>12}")
    print("-" * 60)

    for member_name, data in by_member.items():
        provider = data.get("provider", "unknown")
        model = data.get("model", "unknown")
        input_tokens = data["input"]
        output_tokens = data["output"]
        cost = calculate_cost(provider, model, input_tokens, output_tokens)
        total_cost += cost

        print(f"{member_name:<12} {provider:<12} {input_tokens:>10,} {output_tokens:>10,} ${cost:>10.4f}")

    print("-" * 60)

    totals = session.total_tokens()
    print(f"{'TOTAL':<12} {'':<12} {totals['input']:>10,} {totals['output']:>10,} ${total_cost:>10.4f}")
    print()

    if session.token_usage:
        print("Per-round breakdown:")
        rounds_seen = set()
        for usage in session.token_usage:
            if usage.round_number not in rounds_seen:
                rounds_seen.add(usage.round_number)
                round_tokens = sum(
                    u.total_tokens for u in session.token_usage
                    if u.round_number == usage.round_number
                )
                print(f"  Round {usage.round_number}: {round_tokens:,} tokens")
        print()

    return total_cost


def load_council_config() -> CouncilConfig:
    """Load council configuration from TOML file."""
    import tomllib
    config_path = Path(__file__).parent.parent / "council_config.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Council config not found: {config_path}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return CouncilConfig.from_dict(data.get("council", {}))


def gather_project_state() -> dict:
    """Gather current project state for evidence."""
    state = {}

    # Get test count
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "--collect-only", "-q", "tests/"],
            capture_output=True, text=True, timeout=60
        )
        test_lines = result.stdout.strip().split('\n')
        state["test_count"] = len([l for l in test_lines if l.strip() and not l.startswith('=')])
    except Exception as e:
        state["test_count"] = f"Could not collect: {e}"

    # Get recent commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-15"],
            capture_output=True, text=True, timeout=10
        )
        state["recent_commits"] = result.stdout.strip()
    except Exception:
        state["recent_commits"] = "Could not retrieve git log"

    # Get git status
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=10
        )
        state["git_status"] = result.stdout.strip() or "Working tree clean"
    except Exception:
        state["git_status"] = "Could not retrieve git status"

    # Get branch info
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=10
        )
        state["current_branch"] = result.stdout.strip()
    except Exception:
        state["current_branch"] = "Unknown"

    # Count Python files
    try:
        result = subprocess.run(
            ["find", "kestrel_sovereign", "-name", "*.py", "-type", "f"],
            capture_output=True, text=True, timeout=30
        )
        state["python_file_count"] = len(result.stdout.strip().split('\n'))
    except Exception:
        state["python_file_count"] = "Unknown"

    # Get previous council sessions
    sessions_dir = Path(__file__).parent.parent / "data" / "council_sessions"
    if sessions_dir.exists():
        sessions = list(sessions_dir.glob("*.json"))
        state["previous_sessions"] = len(sessions)
    else:
        state["previous_sessions"] = 0

    return state


def build_progress_review_evidence() -> Evidence:
    """Build the evidence package for progress review."""

    state = gather_project_state()

    evidence = Evidence(
        target="kestrel_sovereign_progress_review",
        code_changes=[
            f"Current Branch: {state['current_branch']}",
            f"Python Files: {state['python_file_count']}",
            f"Git Status: {state['git_status']}",
            "",
            "Recent Commits:",
            state['recent_commits'],
        ],
        test_count=state['test_count'] if isinstance(state['test_count'], int) else 0,
        test_passed=state['test_count'] if isinstance(state['test_count'], int) else 0,
        risks=[
            "Training infrastructure (LoRA) under active development",
            "Some uncommitted changes in training adapters",
            "Emma Genesis conditions from previous council not all verified",
        ],
        architecture_docs=[
            "CLAUDE.md - Agent instructions",
            "council_config.toml - Council configuration",
            "kestrel_sovereign/features/ - 28 feature modules",
        ],
        previous_decisions=[
            f"Previous council sessions: {state['previous_sessions']}",
            "2025-12-27: Emma Genesis APPROVED by unanimous council vote",
            "Key conditions: Safe mode signaling, Day 0 backup, staging drill, 30-day review",
        ],
    )

    return evidence


async def run_council_session():
    """Run the Constitutional Council session for progress review."""

    print("=" * 70)
    print("CONSTITUTIONAL COUNCIL SESSION: PROGRESS REVIEW")
    print("=" * 70)
    print()
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load config
    print("Loading council configuration...")
    config = load_council_config()
    print(f"  Members: {len(config.members)}")
    for member in config.members:
        print(f"    - {member.name} ({member.provider}/{member.model}) - {member.role}")
    print(f"  Consensus Rule: {config.consensus_rule.value}")
    print(f"  Max Rounds: {config.max_rounds}")
    print()

    # Check API keys
    print("Checking API key availability...")
    api_keys = {
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "GOOGLE_APPLICATION_CREDENTIALS": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")),
    }
    for key, available in api_keys.items():
        status = "Available" if available else "Missing"
        print(f"  {key}: {status}")
    print()

    # Build evidence
    print("Compiling evidence package...")
    evidence = build_progress_review_evidence()
    print(f"  Test count: {evidence.test_count}")
    print(f"  Risks identified: {len(evidence.risks)}")
    print(f"  Code changes entries: {len(evidence.code_changes)}")
    print()

    # The question for progress review
    question = """Please review the current state of the Kestrel Sovereign project and provide consensus feedback.

Context:
- Kestrel Sovereign is a Constitutional AI Agent Framework with cryptographic identity (DIDs)
- The project was recently extracted from a larger system and is under active development
- Emma Genesis was APPROVED by the council on 2025-12-27 with specific conditions

Questions for the council:
1. What is your assessment of the project's current technical state?
2. Are the Emma Genesis conditions being adequately addressed?
3. What are the highest priority items that should be focused on next?
4. Are there any constitutional or security concerns with the current direction?
5. What recommendations do you have for the project's evolution?

Please provide comprehensive feedback from your respective roles (constitutional, security, technical)."""

    print("Question for council:")
    print("-" * 50)
    print(question[:500] + "..." if len(question) > 500 else question)
    print("-" * 50)
    print()

    # Convene
    print("Convening council session...")
    print("(This may take 2-5 minutes as each model deliberates)")
    print()

    try:
        session = await convene_council(
            question=question,
            evidence=evidence,
            members=config.members,
            max_rounds=config.max_rounds,
            consensus_rule=config.consensus_rule,
        )

        # Save session
        storage = get_storage()
        await storage.save_session(session)

        # Print results
        print()
        print("=" * 70)
        print("COUNCIL SESSION RESULTS")
        print("=" * 70)
        print()
        print(f"Session ID: {session.id}")
        print(f"Outcome: {session.outcome.value}")
        print(f"Rounds: {len(session.rounds)}")
        print(f"Verdicts: {len(session.verdicts)}")
        print()

        # Print each verdict
        for verdict in session.verdicts:
            emoji = {"APPROVE": "Approved", "REJECT": "Rejected", "ABSTAIN": "Abstained"}.get(verdict.decision.value, "?")
            print(f"[{emoji}] {verdict.member_name}: {verdict.decision.value} ({verdict.confidence:.0%} confidence)")
            print(f"   Reasoning: {verdict.reasoning[:300]}..." if len(verdict.reasoning) > 300 else f"   Reasoning: {verdict.reasoning}")
            if verdict.concerns:
                print(f"   Concerns: {', '.join(verdict.concerns[:3])}")
            if verdict.conditions:
                print(f"   Conditions: {', '.join(verdict.conditions[:3])}")
            print()

        print(f"Full transcript saved to: data/council_sessions/{session.id}.md")

        # Print token usage and costs
        total_cost = print_token_usage(session)
        print(f"Council session cost: ${total_cost:.4f}")
        print()

        return session

    except Exception as e:
        logger.error(f"Council session failed: {e}")
        raise


if __name__ == "__main__":
    session = asyncio.run(run_council_session())

    # Exit with appropriate code
    if session.outcome.value == "APPROVED":
        print("Council session completed - APPROVED")
        sys.exit(0)
    elif session.outcome.value == "REJECTED":
        print("Council session completed - REJECTED")
        sys.exit(1)
    else:
        print("Council session completed - DEADLOCK/PENDING")
        sys.exit(2)
