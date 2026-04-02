"""
Strategic Memory Feature for Kestrel agents.

Provides persistent strategic context (vision, milestones, stakeholders,
decisions, blockers, patterns) that survives across sessions. Loaded from
STRATEGY.yaml in the agent's data directory and injected into the system
prompt via the BootstrapLoader.

This feature also provides !strategy commands for querying and updating
strategic context at runtime.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

from .backlog_hygiene import run_backlog_hygiene
from .morning_signal import generate_morning_signal, generate_portfolio_dashboard
from .session_log import collect_session_log
from .talon_handoff import dispatch_to_talon, pick_top_issue

logger = logging.getLogger(__name__)

# Try to import yaml; fall back to a simple parser if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _load_yaml_simple(text: str) -> dict:
    """Minimal YAML-like loader for when PyYAML is not installed."""
    # This is a fallback -- STRATEGY.yaml is also loaded as raw text
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

    _DEFAULT_TEMPLATE = {
        "version": 1,
        "vision": "Define your agent's long-term vision here.",
        "milestones": [],
        "stakeholders": [],
        "decisions": [],
        "blockers": [],
        "patterns": [],
    }

    def __init__(self, agent):
        super().__init__(agent)
        self._data: Dict[str, Any] = {}
        self._strategy_path: Optional[Path] = None

    @property
    def tool_description(self) -> str:
        return (
            "Query and update the agent's strategic memory -- vision, milestones, "
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
                    # Create default template so the agent can start capturing strategy
                    self._data = dict(self._DEFAULT_TEMPLATE)
                    self._save()
                    if self._strategy_path.exists():
                        logger.info(
                            f"Created default {self.STRATEGY_FILENAME} at {self._strategy_path}"
                        )
                    else:
                        logger.info(
                            f"No {self.STRATEGY_FILENAME} at {self._strategy_path} "
                            "and could not create template -- strategic memory not active"
                        )
            else:
                logger.debug("No agent_data_dir available -- strategic memory not active")

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
                logger.warning("PyYAML not installed -- cannot save strategic memory updates")
                return
            self._strategy_path.write_text(content, encoding="utf-8")
            logger.info("Strategic memory saved")
        except Exception as e:
            logger.error(f"Failed to save strategic memory: {e}")

    # ------------------------------------------------------------------
    # Tools: Strategy CRUD
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
            section: Which section to view -- all, vision, milestones, stakeholders, decisions, blockers, patterns
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
            severity: How severe -- low, medium, high, critical
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
    # Tools: GitHub-powered (delegated to sub-modules)
    # ------------------------------------------------------------------

    @tool(
        name="morning_signal",
        description="Generate a morning strategic briefing -- milestone status, blockers, recommended work items. Pulls live data from GitHub when GITHUB_TOKEN is available.",
        category=ToolCategory.SYSTEM,
        command_prefix="!morning",
    )
    async def morning_signal(self) -> str:
        """Generate the Morning Signal briefing from strategic memory + live GitHub data."""
        return await generate_morning_signal(self._data)

    @tool(
        name="signal_dispatch",
        description="Pick the highest-priority issue from strategic memory and dispatch it to Talon via the Agent Mesh Protocol. Works with any signal source (morning, hygiene, event-driven, on-demand).",
        category=ToolCategory.SYSTEM,
        command_prefix="!dispatch",
    )
    async def signal_dispatch(self, mode: str = "execute") -> str:
        """Pick top issue from strategic memory and dispatch to Talon.

        Args:
            mode: 'execute' to dispatch immediately, 'suggest' to show suggestion only.
        """
        if mode == "suggest":
            issue = await pick_top_issue(self._data)
            if not issue:
                return "## Signal Dispatch (suggest)\nNo actionable issue found."
            return (
                f"## Signal Dispatch (suggest)\n"
                f"**Top issue:** {issue['repo']}#{issue['issue_number']}: {issue['issue_title']}\n"
                f"**Priority:** {issue['priority']}\n"
                f"**Context:** {issue.get('context', 'N/A')}\n\n"
                f"Use `!dispatch` or `!talon claim {issue['repo']} {issue['issue_number']}` to execute."
            )

        # Try TalonCoordinatorFeature first (preferred path)
        coordinator = self._get_talon_coordinator()
        if coordinator:
            issue = await pick_top_issue(self._data)
            if not issue:
                return "## Signal Dispatch\nNo actionable issue found."
            result = await coordinator.talon_claim(
                repo=issue["repo"], issue=issue["issue_number"],
            )
            status = "dispatched" if result.get("dispatched") else f"failed: {result.get('error', 'unknown')}"
            return (
                f"## Signal Dispatch\n"
                f"{issue['repo']}#{issue['issue_number']}: {issue['issue_title']} -- {status}\n"
                f"Method: {result.get('method', 'N/A')}"
            )

        # Fallback: direct mesh dispatch via talon_handoff
        dispatch_result = await dispatch_to_talon(self._data)
        return f"## Signal Dispatch\n{dispatch_result}"

    def _get_talon_coordinator(self):
        """Get TalonCoordinatorFeature if loaded."""
        if hasattr(self.agent, '_features'):
            for f in self.agent._features:
                if type(f).__name__ == "TalonCoordinatorFeature":
                    return f
        return None

    @tool(
        name="portfolio_dashboard",
        description="Open the Portfolio Dashboard -- live operational and strategic intelligence from all repos, milestone tracking, outcome scoreboard, and budget info.",
        category=ToolCategory.SYSTEM,
        command_prefix="!dashboard",
    )
    async def portfolio_dashboard(self) -> str:
        """Generate a link to the Portfolio Dashboard with a live summary."""
        return await generate_portfolio_dashboard(self._data)

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
        return await run_backlog_hygiene(self._data, fix=fix)

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
        return await collect_session_log(self._data, session_id=session_id, focus=focus)

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
            lines.append(f"- **{s.get('name', '?')}**{handle}: {s.get('role', '?')} -- {s.get('context', '')}")
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
