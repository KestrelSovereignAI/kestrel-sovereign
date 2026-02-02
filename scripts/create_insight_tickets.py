#!/usr/bin/env python3
"""
Create GitHub issues from reflection insights.

This script creates GitHub issues directly from insights stored in the database,
bypassing the full constitutional approval flow for development/testing.

Usage:
    python scripts/create_insight_tickets.py agent_data/kestrel_prime.db
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def main():
    # Get database path from args
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_insight_tickets.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    if os.path.isdir(db_path):
        db_path = os.path.join(db_path, "kestrel_prime.db")

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Check GitHub token
    github_token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_PAT or GITHUB_TOKEN environment variable required")
        sys.exit(1)

    # Get target repo
    repo = os.getenv("GITHUB_SELF_REPO", "Kestrel-Sovereign-AI/kestrel")
    print(f"Target repository: {repo}")

    # Import required modules
    import aiosqlite
    from kestrel_sovereign.features.github.client import GitHubClient

    # Initialize GitHub client
    client = GitHubClient(token=github_token)

    # Get insights from database
    async with aiosqlite.connect(db_path) as db:
        # Check if table exists
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reflection_insights'"
        )
        if not await cursor.fetchone():
            print("No reflection_insights table found - run !reflect first")
            sys.exit(1)

        # Get actionable insights
        cursor = await db.execute("""
            SELECT id, type, title, description, evidence, confidence,
                   actionable, suggested_action, created_at
            FROM reflection_insights
            WHERE actionable = 1
            ORDER BY created_at DESC
            LIMIT 10
        """)
        rows = await cursor.fetchall()

        if not rows:
            print("No actionable insights found")
            sys.exit(0)

        print(f"\nFound {len(rows)} actionable insights")

        for row in rows:
            insight_id = row[0]
            insight_type = row[1]
            title = row[2]
            description = row[3]
            evidence = json.loads(row[4]) if row[4] else []
            confidence = row[5]
            suggested_action = row[7]
            created_at = row[8]

            print(f"\n--- Insight: {title[:50]}... ---")
            print(f"    Type: {insight_type}")
            print(f"    Confidence: {confidence:.0%}")

            # Build issue body
            sections = []
            sections.append(f"## {insight_type.title()} Insight")
            sections.append("")
            sections.append(f"**Generated:** {created_at}")
            sections.append(f"**Confidence:** {confidence:.0%}")
            sections.append("")

            sections.append("## Description")
            sections.append("")
            sections.append(description)
            sections.append("")

            if evidence:
                sections.append("## Evidence")
                sections.append("")
                for i, e in enumerate(evidence[:5], 1):
                    sections.append(f"{i}. `{e}`")
                sections.append("")

            if suggested_action:
                sections.append("## Suggested Action")
                sections.append("")
                sections.append(suggested_action)
                sections.append("")

            sections.append("---")
            sections.append("*This issue was created by the Kestrel Agent's reflection system.*")

            body = "\n".join(sections)

            # Determine labels
            labels = ["agent-insight"]
            type_labels = {
                "pattern": "pattern",
                "success": "documentation",
                "failure": "bug",
                "improvement": "enhancement",
                "anomaly": "investigation",
            }
            if insight_type.lower() in type_labels:
                labels.append(type_labels[insight_type.lower()])
            labels.append("actionable")
            if confidence >= 0.9:
                labels.append("high-confidence")

            # Create the issue
            issue_title = f"[Agent Insight] {title}"

            print(f"    Creating issue: {issue_title[:60]}...")

            try:
                result = await client.create_issue(
                    repo=repo,
                    title=issue_title,
                    body=body,
                    labels=labels,
                )
                issue_url = result.get("html_url")
                print(f"    Created: {issue_url}")
            except Exception as e:
                print(f"    ERROR: {e}")
                # Continue with next insight
                continue

    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
