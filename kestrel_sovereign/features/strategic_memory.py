"""
Strategic Memory Feature for Kestrel agents.

Provides persistent strategic context (vision, milestones, stakeholders,
decisions, blockers, patterns) that survives across sessions. Loaded from
STRATEGY.yaml in the agent's data directory and injected into the system
prompt via the BootstrapLoader.

This feature also provides !strategy commands for querying and updating
strategic context at runtime.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)

# Try to import yaml; fall back to a simple parser if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _load_yaml_simple(text: str) -> dict:
    """Minimal YAML-like loader for when PyYAML is not installed."""
    # This is a fallback — STRATEGY.yaml is also loaded as raw text
    # by the BootstrapLoader, so the agent always sees the content.
    return {"_raw": text}


class StrategicMemoryFeature(Feature):
    """
    Feature providing persistent strategic context for Kestrel agents.

    Loads STRATEGY.yaml from the agent's data directory and provides
    tools for querying vision, milestones, stakeholders, decisions,
    blockers, and learned patterns.
    """

    STRATEGY_FILENAME = "STRATEGY.yaml"

    def __init__(self, agent):
        super().__init__(agent)
        self._data: Dict[str, Any] = {}
        self._strategy_path: Optional[Path] = None

    @property
    def tool_description(self) -> str:
        return (
            "Query and update the agent's strategic memory — vision, milestones, "
            "stakeholders, decisions, blockers, and learned patterns"
        )

    async def initialize(self):
        """Load strategic memory from STRATEGY.yaml."""
        try:
            # Find the strategy file in the agent data directory
            agent_data_dir = getattr(self.agent, 'agent_data_dir', None)
            if not agent_data_dir:
                # Try to derive from storage path
                storage_path = getattr(self.agent, 'storage_path', None)
                if storage_path:
                    agent_data_dir = str(Path(storage_path).parent)

            if agent_data_dir:
                self._strategy_path = Path(agent_data_dir) / self.STRATEGY_FILENAME
                if self._strategy_path.exists():
                    raw = self._strategy_path.read_text(encoding="utf-8")
                    if HAS_YAML:
                        self._data = yaml.safe_load(raw) or {}
                    else:
                        self._data = _load_yaml_simple(raw)
                    logger.info(
                        f"StrategicMemoryFeature loaded: {len(self._data)} top-level keys"
                    )
                else:
                    logger.info(
                        f"No {self.STRATEGY_FILENAME} found at {self._strategy_path} — "
                        "strategic memory not active"
                    )
            else:
                logger.debug("No agent_data_dir available — strategic memory not active")

        except Exception as e:
            logger.error(f"Failed to load strategic memory: {e}")

    def _save(self):
        """Persist strategic memory back to STRATEGY.yaml."""
        if not self._strategy_path or not self._data:
            return
        try:
            if HAS_YAML:
                content = yaml.dump(
                    self._data,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    width=100,
                )
            else:
                # Without PyYAML we can't safely round-trip; skip save
                logger.warning("PyYAML not installed — cannot save strategic memory updates")
                return
            self._strategy_path.write_text(content, encoding="utf-8")
            logger.info("Strategic memory saved")
        except Exception as e:
            logger.error(f"Failed to save strategic memory: {e}")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        name="strategy_view",
        description="View the current strategic context: vision, milestones, stakeholders, decisions, blockers, and patterns.",
        category=ToolCategory.SYSTEM,
        command_prefix="!strategy",
    )
    async def strategy_view(self, section: str = "all") -> str:
        """
        View a section of the strategic memory.

        Args:
            section: Which section to view — all, vision, milestones, stakeholders, decisions, blockers, patterns
        """
        if not self._data:
            return "No strategic memory loaded. Create a STRATEGY.yaml in the agent data directory."

        if section == "all":
            return self._format_all()

        if section == "vision":
            return self._data.get("vision", "No vision defined.")

        if section == "milestones":
            return self._format_milestones()

        if section == "stakeholders":
            return self._format_stakeholders()

        if section == "decisions":
            return self._format_decisions()

        if section == "blockers":
            return self._format_blockers()

        if section == "patterns":
            return self._format_patterns()

        return f"Unknown section: {section}. Available: all, vision, milestones, stakeholders, decisions, blockers, patterns"

    @tool(
        name="strategy_add_decision",
        description="Record a new strategic decision to the decision log.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_decision(
        self, decision: str, rationale: str, session: str = "", impact: str = ""
    ) -> str:
        """
        Add a decision to the strategic memory.

        Args:
            decision: What was decided
            rationale: Why it was decided
            session: Session number when the decision was made
            impact: Expected impact of the decision
        """
        if "decisions" not in self._data:
            self._data["decisions"] = []

        entry = {
            "date": str(date.today()),
            "session": session,
            "decision": decision,
            "rationale": rationale,
            "impact": impact,
        }
        self._data["decisions"].append(entry)
        self._save()
        return f"Decision recorded: {decision}"

    @tool(
        name="strategy_add_blocker",
        description="Record a new blocker to the strategic memory.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_blocker(
        self, issue: str, title: str, severity: str = "medium", owner: str = "unassigned", notes: str = ""
    ) -> str:
        """
        Add a blocker to the strategic memory.

        Args:
            issue: Issue number or identifier
            title: Short description of the blocker
            severity: How severe — low, medium, high, critical
            owner: Who owns resolving this blocker
            notes: Additional context
        """
        if "blockers" not in self._data:
            self._data["blockers"] = []

        entry = {
            "issue": issue,
            "title": title,
            "severity": severity,
            "owner": owner,
            "blocked_since": str(date.today()),
            "notes": notes,
        }
        self._data["blockers"].append(entry)
        self._save()
        return f"Blocker recorded: {title}"

    @tool(
        name="strategy_add_pattern",
        description="Record a learned pattern or insight to the strategic memory.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_pattern(self, pattern: str, source: str = "", implication: str = "") -> str:
        """
        Add a learned pattern to the strategic memory.

        Args:
            pattern: The pattern or insight observed
            source: Where this was observed
            implication: What this means for future work
        """
        if "patterns_learned" not in self._data:
            self._data["patterns_learned"] = []

        entry = {
            "pattern": pattern,
            "source": source,
            "implication": implication,
        }
        self._data["patterns_learned"].append(entry)
        self._save()
        return f"Pattern recorded: {pattern}"

    @tool(
        name="strategy_resolve_blocker",
        description="Mark a blocker as resolved and remove it from active blockers.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_resolve_blocker(self, issue: str) -> str:
        """
        Resolve a blocker by its issue identifier.

        Args:
            issue: The issue number or identifier to resolve
        """
        blockers = self._data.get("blockers", [])
        original_count = len(blockers)
        self._data["blockers"] = [b for b in blockers if b.get("issue") != issue]

        if len(self._data["blockers"]) < original_count:
            self._save()
            return f"Blocker {issue} resolved and removed."
        return f"No blocker found with issue: {issue}"

    # ------------------------------------------------------------------
    # GitHub API Integration
    # ------------------------------------------------------------------

    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or .env file."""
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            return token
        # Try .env in project root
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GITHUB_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    async def _github_api(self, path: str, token: str) -> Any:
        """Make a GitHub API GET request. Returns parsed JSON or None on error."""
        url = f"https://api.github.com{path}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "kestrel-agent",
        })
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=10).read(),
            )
            return json.loads(resp)
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            logger.warning(f"GitHub API error for {path}: {e}")
            return None

    @staticmethod
    def _short_repo(repo: str, all_repos: list) -> str:
        """Shorten repo name, using full owner/repo when names collide."""
        name = repo.split("/")[-1]
        if sum(1 for r in all_repos if r.split("/")[-1] == name) > 1:
            return repo
        return name

    async def _fetch_github_signal(self) -> Dict[str, Any]:
        """Fetch live GitHub data for Morning Signal repos.

        Returns a dict with per-repo issue counts, open PRs, and recent comments.
        """
        config = self._data.get("morning_signal_config", {})
        repos = config.get("scan_repos", [])
        if not repos:
            return {}

        token = self._get_github_token()
        if not token:
            logger.info("No GITHUB_TOKEN found — Morning Signal will use YAML data only")
            return {}

        result: Dict[str, Any] = {}
        since = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        for repo in repos:
            repo_data: Dict[str, Any] = {"issues": [], "prs": [], "comments_24h": 0}

            # Fetch open issues (not PRs, up to 100)
            issues = await self._github_api(
                f"/repos/{repo}/issues?state=open&per_page=100", token,
            )
            if isinstance(issues, list):
                # Filter out PRs (they appear in /issues too)
                real_issues = [i for i in issues if "pull_request" not in i]
                repo_data["issue_count"] = len(real_issues)

                # Count by milestone
                by_milestone: Dict[str, int] = {}
                for i in real_issues:
                    ms_name = i.get("milestone", {}).get("title", "No Milestone") if i.get("milestone") else "No Milestone"
                    by_milestone[ms_name] = by_milestone.get(ms_name, 0) + 1
                repo_data["by_milestone"] = by_milestone

                # Collect issues with labels containing "blocked" or severity
                blocked = [
                    {"number": i["number"], "title": i["title"]}
                    for i in real_issues
                    if any("block" in l["name"].lower() for l in i.get("labels", []))
                ]
                repo_data["blocked_issues"] = blocked
            else:
                repo_data["issue_count"] = "?"

            # Fetch open PRs
            prs = await self._github_api(
                f"/repos/{repo}/pulls?state=open&per_page=20", token,
            )
            if isinstance(prs, list):
                repo_data["prs"] = [
                    {
                        "number": p["number"],
                        "title": p["title"],
                        "author": p["user"]["login"],
                        "updated": p["updated_at"][:10],
                    }
                    for p in prs
                ]

            # Fetch recent comments (since midnight UTC)
            comments = await self._github_api(
                f"/repos/{repo}/issues/comments?since={since}&sort=updated&direction=desc&per_page=5",
                token,
            )
            if isinstance(comments, list):
                repo_data["comments_24h"] = len(comments)
                repo_data["recent_comments"] = [
                    {
                        "issue_url": c.get("issue_url", "").split("/")[-1],
                        "author": c["user"]["login"],
                        "snippet": (c.get("body") or "")[:80],
                    }
                    for c in comments[:3]
                ]

            result[repo] = repo_data

        return result

    # ------------------------------------------------------------------
    # Morning Signal
    # ------------------------------------------------------------------

    @tool(
        name="morning_signal",
        description="Generate a morning strategic briefing — milestone status, blockers, recommended work items. Pulls live data from GitHub when GITHUB_TOKEN is available.",
        category=ToolCategory.SYSTEM,
        command_prefix="!morning",
    )
    async def morning_signal(self) -> str:
        """Generate the Morning Signal briefing from strategic memory + live GitHub data."""
        if not self._data:
            return "No strategic memory loaded. Create a STRATEGY.yaml first."

        today = date.today()
        lines = [f"# Morning Signal — {today.strftime('%B %d, %Y')}", ""]

        # Fetch live GitHub data (non-blocking, graceful on failure)
        github_data = await self._fetch_github_signal()
        has_live = bool(github_data)

        if has_live:
            lines.append("*Live data from GitHub*")
            lines.append("")

        # Repo overview (live data)
        if has_live:
            lines.append("## Repo Overview")
            lines.append("| Repo | Open Issues | Open PRs | Activity (24h) |")
            lines.append("|------|-------------|----------|----------------|")
            total_issues = 0
            total_prs = 0
            total_comments = 0
            for repo, data in github_data.items():
                ic = data.get("issue_count", "?")
                pc = len(data.get("prs", []))
                cc = data.get("comments_24h", 0)
                short_repo = self._short_repo(repo, list(github_data.keys()))
                lines.append(f"| {short_repo} | {ic} | {pc} | {cc} comments |")
                if isinstance(ic, int):
                    total_issues += ic
                total_prs += pc
                total_comments += cc
            lines.append(f"| **Total** | **{total_issues}** | **{total_prs}** | **{total_comments}** |")
            lines.append("")

        # Milestones (YAML + enriched with live counts)
        lines.append("## Milestones")
        for m in self._data.get("milestones", []):
            status_icon = {
                "on_track": "ON TRACK",
                "in_progress": "IN PROGRESS",
                "at_risk": "AT RISK",
                "complete": "COMPLETE",
                "data_collection": "COLLECTING DATA",
                "active": "ACTIVE",
            }.get(m.get("status", ""), m.get("status", "unknown"))

            due = m.get("due", "no date")
            if due and str(due) not in ("no date", "ongoing", "null", "None"):
                try:
                    due_date = datetime.strptime(str(due), "%Y-%m-%d").date()
                    days_left = (due_date - today).days
                    due_str = f"due {due} ({days_left} days)"
                except (ValueError, TypeError):
                    due_str = f"due: {due}"
            else:
                due_str = str(due) if due else "no date"

            owner = m.get("owner", "unassigned")

            # Enrich with live issue count from GitHub
            live_count = None
            milestone_name = m.get("name", "")
            if has_live:
                for repo in m.get("repos", []):
                    rd = github_data.get(repo, {})
                    by_ms = rd.get("by_milestone", {})
                    for ms_key, count in by_ms.items():
                        if milestone_name.lower() in ms_key.lower():
                            live_count = (live_count or 0) + count

            open_str = f"{live_count} open issues" if live_count is not None else f"{m.get('open_items', '?')} open items"
            lines.append(f"- **{milestone_name}**: [{status_icon}] {due_str}, owner: {owner}, {open_str}")

            cp = m.get("critical_path")
            if cp:
                lines.append(f"  - Critical path: {cp}")

        # Open PRs (live data)
        if has_live:
            all_prs = []
            all_repos = list(github_data.keys())
            for repo, data in github_data.items():
                for pr in data.get("prs", []):
                    pr["repo"] = self._short_repo(repo, all_repos)
                    all_prs.append(pr)
            if all_prs:
                lines.append("")
                lines.append("## Open Pull Requests")
                for pr in all_prs:
                    lines.append(f"- **{pr['repo']}#{pr['number']}**: {pr['title']} (@{pr['author']}, updated {pr['updated']})")

        # Blockers (YAML + live blocked issues)
        blockers = self._data.get("blockers", [])
        live_blocked = []
        if has_live:
            all_repos = list(github_data.keys())
            for repo, data in github_data.items():
                for b in data.get("blocked_issues", []):
                    live_blocked.append({"repo": self._short_repo(repo, all_repos), **b})

        if blockers or live_blocked:
            lines.append("")
            lines.append("## Blockers")
            for b in blockers:
                severity = b.get("severity", "?").upper()
                since = b.get("blocked_since", "?")
                lines.append(f"- [{severity}] {b.get('issue', '?')}: {b.get('title', '?')} (since {since}, owner: {b.get('owner', 'unassigned')})")
                notes = b.get("notes")
                if notes:
                    lines.append(f"  - {notes}")
            # Add any GitHub-labeled blocked issues not already in YAML
            yaml_issues = {b.get("issue", "").replace("#", "") for b in blockers}
            for lb in live_blocked:
                if str(lb["number"]) not in yaml_issues:
                    lines.append(f"- [GITHUB] {lb['repo']}#{lb['number']}: {lb['title']}")

        # Recent activity highlights (live data)
        if has_live:
            active_repos = [
                (repo, data) for repo, data in github_data.items()
                if data.get("comments_24h", 0) > 0
            ]
            if active_repos:
                lines.append("")
                lines.append("## Recent Activity")
                for repo, data in active_repos:
                    short = self._short_repo(repo, list(github_data.keys()))
                    lines.append(f"- **{short}**: {data['comments_24h']} comments today")
                    for c in data.get("recent_comments", []):
                        lines.append(f"  - #{c['issue_url']} (@{c['author']}): {c['snippet']}...")

        # Suggested work items (based on milestones + blockers)
        lines.append("")
        lines.append("## Suggested Work Items (by impact)")
        suggestions = []

        # Suggest unblocking blockers first
        for b in blockers:
            if b.get("severity") in ("high", "critical"):
                suggestions.append(f"Unblock {b.get('issue', '?')}: {b.get('title', '?')}")

        # Suggest work on in-progress milestones
        for m in self._data.get("milestones", []):
            if m.get("status") in ("in_progress", "on_track", "at_risk"):
                cp = m.get("critical_path")
                if cp:
                    suggestions.append(f"{m.get('name', '?')}: {cp}")

        # Suggest reviewing open PRs
        if has_live and all_prs:
            suggestions.append(f"Review {len(all_prs)} open PR(s)")

        for i, s in enumerate(suggestions[:5], 1):
            lines.append(f"{i}. {s}")

        if not suggestions:
            lines.append("No urgent items detected. Review backlog for next priorities.")

        lines.append("")
        if not has_live:
            lines.append("*Set GITHUB_TOKEN to enable live repo scanning.*")
            lines.append("")
        lines.append("**Your call. What resonates?**")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Portfolio Dashboard
    # ------------------------------------------------------------------

    @tool(
        name="portfolio_dashboard",
        description="Open the Portfolio Dashboard — live operational and strategic intelligence from all repos, milestone tracking, outcome scoreboard, and budget info.",
        category=ToolCategory.SYSTEM,
        command_prefix="!dashboard",
    )
    async def portfolio_dashboard(self) -> str:
        """Generate a link to the Portfolio Dashboard with a live summary."""
        config = self._data.get("morning_signal_config", {})
        repos = config.get("scan_repos", [])

        # Try to detect the host port from environment or default
        host_port = os.environ.get("KESTREL_HOST_PORT", "8888")

        lines = ["# Portfolio Dashboard", ""]
        lines.append(f"**Open in browser:** http://localhost:{host_port}/static/dashboard.html")
        lines.append("")

        # Quick summary from live data
        token = self._get_github_token()
        if token and repos:
            total_issues = 0
            total_prs = 0
            for repo in repos:
                issues = await self._github_api(
                    f"/repos/{repo}/issues?state=open&per_page=100", token,
                )
                if isinstance(issues, list):
                    real = [i for i in issues if "pull_request" not in i]
                    total_issues += len(real)

                prs = await self._github_api(
                    f"/repos/{repo}/pulls?state=open&per_page=20", token,
                )
                if isinstance(prs, list):
                    total_prs += len(prs)

            lines.append(f"**Quick snapshot:** {total_issues} open issues, {total_prs} open PRs across {len(repos)} repos")
        else:
            lines.append("*Set GITHUB_TOKEN to see live data in the dashboard.*")

        lines.append("")
        lines.append("The dashboard has 4 tabs:")
        lines.append("1. **Operational** — open issues, PRs, blockers, backlog health, activity")
        lines.append("2. **Strategic** — milestones, velocity trend, decision log")
        lines.append("3. **Scoreboard** — outcome rankings by contributor")
        lines.append("4. **Budget** — engineering cost savings, ceremony cost avoided")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backlog Hygiene Bot
    # ------------------------------------------------------------------

    @tool(
        name="backlog_hygiene",
        description="Scan all repos for backlog hygiene issues: missing assignees, milestones, status labels. Reports gaps and flags items needing human review.",
        category=ToolCategory.SYSTEM,
        command_prefix="!hygiene",
    )
    async def backlog_hygiene(self, fix: str = "no") -> str:
        """Scan repos for backlog hygiene issues and optionally auto-fix.

        Args:
            fix: Set to 'yes' to auto-fix issues where possible (add labels). Default 'no' (report only).
        """
        config = self._data.get("morning_signal_config", {})
        repos = config.get("scan_repos", [])
        if not repos:
            return "No scan_repos configured in morning_signal_config."

        token = self._get_github_token()
        if not token:
            return "No GITHUB_TOKEN found. Set GITHUB_TOKEN environment variable or add to .env file."

        auto_fix = fix.lower() in ("yes", "true", "1")
        today_str = date.today().strftime("%B %d, %Y")
        lines = [f"# Backlog Hygiene Report — {today_str}", ""]

        total_issues = 0
        missing_assignee: List[Dict] = []
        missing_milestone: List[Dict] = []
        missing_labels: List[Dict] = []
        stale_issues: List[Dict] = []
        needs_review: List[Dict] = []
        fixes_applied: List[str] = []

        all_repos = repos[:]

        for repo in repos:
            issues = await self._github_api(
                f"/repos/{repo}/issues?state=open&per_page=100", token,
            )
            if not isinstance(issues, list):
                lines.append(f"- **{repo}**: API error, skipped")
                continue

            real_issues = [i for i in issues if "pull_request" not in i]
            short = self._short_repo(repo, all_repos)
            total_issues += len(real_issues)

            for issue in real_issues:
                num = issue["number"]
                title = issue["title"]
                labels = [l["name"] for l in issue.get("labels", [])]
                assignee = issue.get("assignee")
                milestone = issue.get("milestone")
                updated = issue.get("updated_at", "")[:10]

                entry = {
                    "repo": repo,
                    "short_repo": short,
                    "number": num,
                    "title": title,
                    "labels": labels,
                    "updated": updated,
                }

                # Check: missing assignee
                if not assignee:
                    missing_assignee.append(entry)

                # Check: missing milestone
                if not milestone:
                    missing_milestone.append(entry)

                # Check: missing status label
                status_labels = {"status:blocked", "status:in-progress",
                                 "status:ready", "status:done", "status:backlog"}
                has_status = any(l.lower() in status_labels or
                                l.lower().startswith("status:") for l in labels)
                has_any_label = len(labels) > 0
                if not has_any_label:
                    missing_labels.append(entry)

                # Check: stale (not updated in 14+ days)
                if updated:
                    try:
                        updated_date = datetime.strptime(updated, "%Y-%m-%d").date()
                        days_stale = (date.today() - updated_date).days
                        if days_stale >= 14:
                            entry["days_stale"] = days_stale
                            stale_issues.append(entry)
                    except ValueError:
                        pass

                # Flag for human review: no assignee AND no milestone
                if not assignee and not milestone:
                    needs_review.append(entry)

        # Build report
        lines.append("## Summary")
        lines.append(f"- **{total_issues}** open issues across {len(repos)} repos")
        lines.append(f"- **{len(missing_assignee)}** missing assignee")
        lines.append(f"- **{len(missing_milestone)}** missing milestone")
        lines.append(f"- **{len(missing_labels)}** missing labels")
        lines.append(f"- **{len(stale_issues)}** stale (14+ days without update)")
        lines.append(f"- **{len(needs_review)}** need human review (no assignee + no milestone)")
        lines.append("")

        if missing_assignee:
            lines.append("## Missing Assignee")
            for e in missing_assignee:
                lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
            lines.append("")

        if missing_milestone:
            lines.append("## Missing Milestone")
            for e in missing_milestone:
                lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
            lines.append("")

        if missing_labels:
            lines.append("## Missing Labels")
            for e in missing_labels:
                lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")

            # Auto-fix: add 'needs-triage' label
            if auto_fix:
                lines.append("")
                lines.append("### Auto-fix: Adding 'needs-triage' label")
                for e in missing_labels:
                    result = await self._github_api_post(
                        f"/repos/{e['repo']}/issues/{e['number']}/labels",
                        token,
                        {"labels": ["needs-triage"]},
                    )
                    if result is not None:
                        fixes_applied.append(
                            f"Added 'needs-triage' to {e['short_repo']}#{e['number']}"
                        )
                        lines.append(f"  - Added to {e['short_repo']}#{e['number']}")
                    else:
                        lines.append(f"  - FAILED: {e['short_repo']}#{e['number']}")
            lines.append("")

        if stale_issues:
            lines.append("## Stale Issues (14+ days)")
            for e in stale_issues:
                lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']} ({e['days_stale']} days)")
            lines.append("")

        if needs_review:
            lines.append("## Needs Human Review")
            lines.append("*These issues have no assignee AND no milestone — the AI agent cannot determine who owns them or which workstream they belong to.*")
            lines.append("")
            for e in needs_review:
                lines.append(f"- {e['short_repo']}#{e['number']}: {e['title']}")
            lines.append("")

        # Health score
        if total_issues > 0:
            issues_ok = total_issues - len(missing_assignee) - len(missing_milestone)
            health_pct = round((issues_ok / total_issues) * 100)
            lines.append(f"## Backlog Health Score: {health_pct}%")
            if health_pct >= 90:
                lines.append("Backlog is clean.")
            elif health_pct >= 70:
                lines.append("Backlog needs minor cleanup.")
            else:
                lines.append("Backlog needs attention — too many unowned or untracked issues.")
        lines.append("")

        if fixes_applied:
            lines.append(f"**{len(fixes_applied)} auto-fixes applied.**")
        elif auto_fix:
            lines.append("**No auto-fixes were needed.**")
        else:
            lines.append("*Run `!hygiene fix=yes` to auto-apply labels where possible.*")

        return "\n".join(lines)

    async def _github_api_post(self, path: str, token: str, body: dict) -> Any:
        """Make a GitHub API POST request. Returns parsed JSON or None on error."""
        url = f"https://api.github.com{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "kestrel-agent",
        })
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=10).read(),
            )
            return json.loads(resp)
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            logger.warning(f"GitHub API POST error for {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # End-of-Day Session Log Collector
    # ------------------------------------------------------------------

    @tool(
        name="session_log",
        description="End-of-day session log collector. Scans all repos for today's activity (issues closed, PRs merged, comments, commits) and generates a structured session summary with outcomes and metrics.",
        category=ToolCategory.SYSTEM,
        command_prefix="!sessionlog",
    )
    async def session_log(self, session_id: str = "", focus: str = "") -> str:
        """Collect end-of-day session log from GitHub activity.

        Args:
            session_id: Session number (e.g. '020'). Auto-generated if empty.
            focus: Brief description of today's focus area.
        """
        config = self._data.get("morning_signal_config", {})
        repos = config.get("scan_repos", [])
        if not repos:
            return "No scan_repos configured in morning_signal_config."

        token = self._get_github_token()
        if not token:
            return "No GITHUB_TOKEN found. Set GITHUB_TOKEN environment variable or add to .env file."

        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        since = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Collect data across all repos
        issues_closed: List[Dict] = []
        prs_merged: List[Dict] = []
        prs_opened: List[Dict] = []
        comments_posted: List[Dict] = []
        all_repos = repos[:]
        contributors: Dict[str, Dict] = {}

        for repo in repos:
            short = self._short_repo(repo, all_repos)

            # Closed issues today
            closed = await self._github_api(
                f"/repos/{repo}/issues?state=closed&since={since}&per_page=50&sort=updated", token,
            )
            if isinstance(closed, list):
                for i in closed:
                    if "pull_request" not in i and i.get("closed_at", "")[:10] == today_str:
                        closer = (i.get("closed_by") or i.get("user") or {}).get("login", "unknown")
                        issues_closed.append({
                            "repo": short, "number": i["number"],
                            "title": i["title"], "closer": closer,
                        })
                        contributors.setdefault(closer, {"closed": 0, "prs": 0, "comments": 0})
                        contributors[closer]["closed"] += 1

            # Merged PRs today
            merged = await self._github_api(
                f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=20", token,
            )
            if isinstance(merged, list):
                for p in merged:
                    if p.get("merged_at") and p["merged_at"][:10] == today_str:
                        author = (p.get("user") or {}).get("login", "unknown")
                        prs_merged.append({
                            "repo": short, "number": p["number"],
                            "title": p["title"], "author": author,
                        })
                        contributors.setdefault(author, {"closed": 0, "prs": 0, "comments": 0})
                        contributors[author]["prs"] += 1

            # PRs opened today
            opened = await self._github_api(
                f"/repos/{repo}/pulls?state=open&sort=created&direction=desc&per_page=10", token,
            )
            if isinstance(opened, list):
                for p in opened:
                    if p.get("created_at", "")[:10] == today_str:
                        author = (p.get("user") or {}).get("login", "unknown")
                        prs_opened.append({
                            "repo": short, "number": p["number"],
                            "title": p["title"], "author": author,
                        })

            # Comments today
            comments = await self._github_api(
                f"/repos/{repo}/issues/comments?since={since}&sort=updated&direction=desc&per_page=30",
                token,
            )
            if isinstance(comments, list):
                for c in comments:
                    author = (c.get("user") or {}).get("login", "unknown")
                    issue_num = (c.get("issue_url") or "").split("/")[-1]
                    comments_posted.append({
                        "repo": short, "issue": issue_num, "author": author,
                        "snippet": (c.get("body") or "")[:60],
                    })
                    contributors.setdefault(author, {"closed": 0, "prs": 0, "comments": 0})
                    contributors[author]["comments"] += 1

        # Build session report
        sid = session_id or today.strftime("%Y%m%d")
        lines = [
            f"# Session Log — {today.strftime('%B %d, %Y')} (#{sid})",
            "",
        ]
        if focus:
            lines.append(f"**Focus:** {focus}")
            lines.append("")

        # Outcomes summary
        lines.append("## Outcomes")
        lines.append(f"- **{len(issues_closed)}** issues closed")
        lines.append(f"- **{len(prs_merged)}** PRs merged")
        lines.append(f"- **{len(prs_opened)}** PRs opened")
        lines.append(f"- **{len(comments_posted)}** comments posted")
        lines.append("")

        # Issues closed
        if issues_closed:
            lines.append("## Issues Closed")
            for i in issues_closed:
                lines.append(f"- {i['repo']}#{i['number']}: {i['title']} (@{i['closer']})")
            lines.append("")

        # PRs merged
        if prs_merged:
            lines.append("## PRs Merged")
            for p in prs_merged:
                lines.append(f"- {p['repo']}#{p['number']}: {p['title']} (@{p['author']})")
            lines.append("")

        # PRs opened
        if prs_opened:
            lines.append("## PRs Opened")
            for p in prs_opened:
                lines.append(f"- {p['repo']}#{p['number']}: {p['title']} (@{p['author']})")
            lines.append("")

        # Contributor scoreboard
        if contributors:
            lines.append("## Contributor Scoreboard")
            lines.append("| Contributor | Issues Closed | PRs Merged | Comments | Score |")
            lines.append("|------------|--------------|------------|----------|-------|")
            sorted_c = sorted(
                contributors.items(),
                key=lambda x: x[1]["closed"] + x[1]["prs"] * 2,
                reverse=True,
            )
            for name, counts in sorted_c:
                score = counts["closed"] + counts["prs"] * 2
                lines.append(
                    f"| @{name} | {counts['closed']} | {counts['prs']} | {counts['comments']} | {score} |"
                )
            lines.append("")

        # Activity feed (last 5 comments)
        if comments_posted:
            lines.append("## Activity Highlights")
            for c in comments_posted[:5]:
                lines.append(f"- {c['repo']}#{c['issue']} (@{c['author']}): {c['snippet']}...")
            lines.append("")

        if not issues_closed and not prs_merged and not comments_posted:
            lines.append("*No GitHub activity detected today. Activity may be happening locally.*")
            lines.append("")

        lines.append(f"*Auto-collected at {datetime.now().strftime('%H:%M')} from {len(repos)} repos.*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def _format_all(self) -> str:
        sections = []
        vision = self._data.get("vision", "")
        if vision:
            sections.append(f"## Vision\n{vision}")
        sections.append(self._format_milestones())
        sections.append(self._format_stakeholders())
        sections.append(self._format_decisions())
        sections.append(self._format_blockers())
        sections.append(self._format_patterns())
        return "\n\n".join(sections)

    def _format_milestones(self) -> str:
        milestones = self._data.get("milestones", [])
        if not milestones:
            return "## Milestones\nNone defined."
        lines = ["## Milestones"]
        for m in milestones:
            lines.append(f"- **{m.get('name', '?')}** [{m.get('status', '?')}] due: {m.get('due', '?')}, owner: {m.get('owner', '?')}")
            lines.append(f"  {m.get('summary', '')}")
        return "\n".join(lines)

    def _format_stakeholders(self) -> str:
        stakeholders = self._data.get("stakeholders", [])
        if not stakeholders:
            return "## Stakeholders\nNone defined."
        lines = ["## Stakeholders"]
        for s in stakeholders:
            handle = f" (@{s['handle']})" if s.get("handle") else ""
            lines.append(f"- **{s.get('name', '?')}**{handle}: {s.get('role', '?')} — {s.get('context', '')}")
        return "\n".join(lines)

    def _format_decisions(self) -> str:
        decisions = self._data.get("decisions", [])
        if not decisions:
            return "## Decision Log\nNo decisions recorded."
        lines = ["## Decision Log"]
        for d in decisions:
            lines.append(f"- [{d.get('date', '?')}] {d.get('decision', '?')}")
            lines.append(f"  Rationale: {d.get('rationale', '')}")
        return "\n".join(lines)

    def _format_blockers(self) -> str:
        blockers = self._data.get("blockers", [])
        if not blockers:
            return "## Blockers\nNo active blockers."
        lines = ["## Blockers"]
        for b in blockers:
            lines.append(f"- [{b.get('severity', '?').upper()}] {b.get('issue', '?')}: {b.get('title', '?')} (owner: {b.get('owner', 'unassigned')})")
        return "\n".join(lines)

    def _format_patterns(self) -> str:
        patterns = self._data.get("patterns_learned", [])
        if not patterns:
            return "## Patterns Learned\nNone recorded."
        lines = ["## Patterns Learned"]
        for p in patterns:
            lines.append(f"- {p.get('pattern', '?')}")
            impl = p.get("implication")
            if impl:
                lines.append(f"  Implication: {impl}")
        return "\n".join(lines)
