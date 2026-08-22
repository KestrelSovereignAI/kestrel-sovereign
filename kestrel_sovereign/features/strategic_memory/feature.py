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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.enum_coerce import normalize_choice as _normalize_choice

from .backlog_hygiene import is_auto_fix, run_backlog_hygiene
from .decision_index import decision_entries, project_decisions
from .issue_selection import pick_top_issue
from .morning_signal import generate_morning_signal
from .session_log import collect_session_log

logger = logging.getLogger(__name__)


# Synonyms LLMs reach for on the strategic-memory enums. severity's canonical
# middle value is ``medium`` here (unlike todo priority's ``normal``), so the
# alias map is local.
_SEVERITY_ALIASES = {
    "moderate": "medium", "med": "medium", "normal": "medium",
    "crit": "critical", "urgent": "critical", "blocker": "critical",
    # NOTE: deliberately no "sevN" aliases — Sev1 means "most critical" in some
    # incident taxonomies and "high" in others; too ambiguous to guess, so they
    # fall through to a value-listing error.
}
# Asymmetric on purpose: ``signal_dispatch`` can start real work, so only
# preview-like synonyms normalize onto the safe ``suggest`` side.  A live
# dispatch still requires the literal ``execute`` value.
_DISPATCH_MODE_ALIASES = {
    "dry-run": "suggest", "dryrun": "suggest", "dry_run": "suggest",
    "preview": "suggest", "plan": "suggest", "simulate": "suggest",
    "suggestion": "suggest", "propose": "suggest",
}

# Strategic Memory knows a stable workflow capability name, never its provider
# class.  An independently installed feature contributes this registration to
# the agent's operator registry; a generic workflow runner exposes
# ``workflow_run``.  Both must be live before execute mode is admitted.
_DEFAULT_DISPATCH_WORKFLOW = "fleet_coding_pipeline"
_WORKFLOW_RUN_TOOL = "workflow_run"
# Prefixes that the github-backed sub-modules (backlog_hygiene,
# session_log) return when prerequisites (scan_repos config or
# GITHUB_TOKEN) are missing. They look like report bodies but are
# actually skipped runs — the @tool wrappers must turn these into
# ERROR envelopes so callers can't treat a no-op as a successful
# scan/log/apply.
_GITHUB_PREREQ_FAILURE_PREFIXES: tuple = (
    "No scan_repos configured",
    "No GITHUB_TOKEN found",
)


def _is_github_prereq_failure(body: str) -> bool:
    return any(body.startswith(p) for p in _GITHUB_PREREQ_FAILURE_PREFIXES)

# Try to import yaml; fall back to a simple parser if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class _SaveOutcome:
    """Truthful result of a ``_save()`` attempt.

    Distinguishes three states so mutating tools can report honestly
    (F291): the write actually persisted, the feature is not active
    (no strategy path — nothing could be saved), or the in-memory
    update happened but the write itself failed.
    """

    persisted: bool
    no_path: bool = False
    error: Optional[str] = None


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

    @property
    def promote_tools_on_startup(self) -> bool:
        # #1578 (B): the decision-log/strategy tools (strategy_add_decision,
        # strategy_open_decisions, etc.) are part of an agent's
        # operational memory loop. Forcing a subagent-dispatch hop
        # before they're advertised means an agent trying to record
        # a strategic decision on turn 1 hits "not advertised" —
        # exactly the failure Emma surfaced. Pinned by
        # _promote_startup_feature_tools so LRU eviction can't
        # silently drop them (#1580 / D).
        return True

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

        # Rebuild the graph index from what YAML says (#2851). Doing it at load
        # is what makes the index genuinely derived: an agent whose database was
        # lost, or whose STRATEGY.yaml was edited by hand or pulled from git,
        # gets a correct index on next start with no migration step. It is also
        # how decisions recorded before this existed become reachable at all.
        await self._reindex_decisions()

    def _projection_agent_id(self) -> Optional[str]:
        """The identity decisions are indexed under.

        Read off the AGENT. The feature has no ``agent_id`` of its own — this
        previously read ``self.agent_id``, which no feature defines, so every
        projection raised AttributeError into the best-effort handler below and
        the index never populated in production at all. The tests did not catch
        it because they assigned ``feature.agent_id`` by hand, so the mutants
        died against a world that only existed in the tests (#2851).
        """
        for attribute in ("agent_id", "did", "id"):
            value = getattr(self.agent, attribute, None)
            if isinstance(value, str) and value.strip():
                return value
        return None

    async def _reindex_decisions(self) -> Dict[str, Any]:
        """Project STRATEGY.yaml decisions into the graph.

        Never raises and never touches YAML: the canonical record is already on
        disk, so a graph failure must not turn into a strategic-memory failure.

        Reconciles rather than merely upserting. YAML is canonical, so a
        decision removed or edited there must stop being reachable through the
        index — otherwise ``recall_decisions`` returns decisions absent from the
        canonical file, which is the opposite of a derived index.
        """
        agent_id = self._projection_agent_id()
        if not agent_id:
            logger.warning(
                "decision index skipped: agent exposes no agent_id/did/id"
            )
            return {"projected": 0, "skipped": 0, "failed": 0,
                    "skipped_reason": "no_agent_identity"}
        entries = decision_entries(self._data)
        try:
            storage = getattr(self.agent, "storage", None)
            graph_store = getattr(storage, "graph", None) if storage else None
            report = await project_decisions(graph_store, agent_id, entries)
        except Exception as e:  # noqa: BLE001 - the index is best-effort
            logger.warning("decision index projection failed: %s", e)
            return {"projected": 0, "skipped": 0, "failed": len(entries)}
        if report.get("failed") or report.get("skipped_reason"):
            logger.info("decision index: %s", report)
        return report

    def _save(self) -> _SaveOutcome:
        """Persist strategic memory back to STRATEGY.yaml.

        Returns a :class:`_SaveOutcome` describing what actually
        happened rather than swallowing failures. Callers must not
        report success without checking it (F291): a missing strategy
        path means the feature is not active and nothing can be saved,
        and a write exception means the in-memory state diverged from
        disk.
        """
        if not self._strategy_path:
            return _SaveOutcome(
                persisted=False,
                no_path=True,
                error=(
                    "No strategy path configured -- strategic memory is not "
                    "active, so nothing was persisted."
                ),
            )
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
                # Without PyYAML we can't safely round-trip.
                msg = "PyYAML not installed -- cannot save strategic memory updates."
                logger.warning(msg)
                return _SaveOutcome(persisted=False, error=msg)
            self._strategy_path.write_text(content, encoding="utf-8")
            logger.info("Strategic memory saved")
            return _SaveOutcome(persisted=True)
        except Exception as e:
            logger.error(f"Failed to save strategic memory: {e}")
            return _SaveOutcome(
                persisted=False,
                error=f"Failed to save strategic memory: {e}",
            )

    def _persisted_result(self, confirmation: str, data: Dict[str, Any]) -> ToolResult:
        """Turn a ``_save()`` outcome into an honest ToolResult (F291).

        - No strategy path -> ERROR (nothing could be persisted).
        - Write failed -> PARTIAL (in-memory state updated, not on disk).
        - Persisted -> OK.
        """
        outcome = self._save()
        if outcome.no_path:
            return ToolResult.failed(
                outcome.error
                or "No strategy path configured; nothing was persisted.",
                data={**data, "persisted": False},
            )
        if not outcome.persisted:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    outcome.error
                    or "In-memory update applied but the write failed; "
                    "changes were not persisted."
                ),
                data={**data, "persisted": False},
            )
        return ToolResult.ok(
            confirmation=confirmation,
            data={**data, "persisted": True},
        )

    # ------------------------------------------------------------------
    # Tools: Strategy CRUD
    # ------------------------------------------------------------------

    @tool(
        name="strategy_view",
        description="View the current strategic context: vision, milestones, stakeholders, decisions, blockers, and patterns.",
        category=ToolCategory.SYSTEM,
        command_prefix="!strategy",
    )
    async def strategy_view(self, section: str = "all") -> ToolResult:
        """
        View a section of the strategic memory.

        Args:
            section: Which section to view -- all, vision, milestones, stakeholders, decisions, blockers, patterns
        """
        if not self._data:
            return ToolResult.failed(
                "No strategic memory loaded. Create a STRATEGY.yaml in the agent data directory.",
            )

        section_renderers = {
            "all": self._format_all,
            # `or` not `default=` — ``vision: ""`` and ``vision: null``
            # both return empty/None from .get; ToolResult.ok requires
            # a non-empty confirmation, so fall back to the placeholder
            # whenever the value is falsy, not just missing.
            "vision": lambda: self._data.get("vision") or "No vision defined.",
            "milestones": self._format_milestones,
            "stakeholders": self._format_stakeholders,
            "decisions": self._format_decisions,
            "blockers": self._format_blockers,
            "patterns": self._format_patterns,
        }
        normalized_section = section.strip().lower() if isinstance(section, str) else section
        renderer = section_renderers.get(normalized_section)
        if renderer is None:
            return ToolResult.failed(
                f"Unknown section: {section}. Available: "
                + ", ".join(section_renderers.keys()),
                data={"section": section},
            )
        section = normalized_section
        body = renderer()
        return ToolResult.ok(
            confirmation=body,
            data={"section": section, "body": body},
        )

    @tool(
        name="strategy_add_decision",
        description="Record a new strategic decision to the decision log.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_decision(
        self, decision: str, rationale: str, session: str = "", impact: str = ""
    ) -> ToolResult:
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
        result = self._persisted_result(
            confirmation=f"Decision recorded: {decision}",
            data={"recorded": True, "decision": entry},
        )
        # Index it so recall_decisions and mark_superseded can see it — but
        # only if it reached canonical YAML. The entry is in ``self._data``
        # whether or not the write succeeded, so projecting unconditionally
        # would publish a decision through the index that does not exist in the
        # file the index is derived from. The projection still cannot change
        # the outcome: a decision that reached disk was recorded whether or not
        # its index entry landed (#2851).
        if (result.data or {}).get("persisted"):
            await self._reindex_decisions()
        return result

    @tool(
        name="strategy_add_blocker",
        description="Record a new blocker to the strategic memory.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_blocker(
        self, issue: str, title: str, severity: str = "medium", owner: str = "unassigned", notes: str = ""
    ) -> ToolResult:
        """
        Add a blocker to the strategic memory.

        Args:
            issue: Issue number or identifier
            title: Short description of the blocker
            severity: How severe -- one of low, medium, high, critical (default medium)
            owner: Who owns resolving this blocker
            notes: Additional context
        """
        # Normalize + validate so an unrecognized severity isn't persisted
        # verbatim (later sort/format code only understands the four levels).
        severity = _normalize_choice(severity or "", _SEVERITY_ALIASES)
        if severity not in ("low", "medium", "high", "critical"):
            return ToolResult.failed(
                f"Invalid severity '{severity}'. Must be one of: low, medium, high, critical.",
                data={"severity": severity},
            )
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
        return self._persisted_result(
            confirmation=f"Blocker recorded: {title}",
            data={"recorded": True, "blocker": entry},
        )

    @tool(
        name="strategy_add_pattern",
        description="Record a learned pattern or insight to the strategic memory.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_add_pattern(self, pattern: str, source: str = "", implication: str = "") -> ToolResult:
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
        return self._persisted_result(
            confirmation=f"Pattern recorded: {pattern}",
            data={"recorded": True, "pattern": entry},
        )

    @tool(
        name="strategy_resolve_blocker",
        description="Mark a blocker as resolved and remove it from active blockers.",
        category=ToolCategory.SYSTEM,
    )
    async def strategy_resolve_blocker(self, issue: str) -> ToolResult:
        """
        Resolve a blocker by its issue identifier.

        Args:
            issue: The issue number or identifier to resolve
        """
        blockers = self._data.get("blockers", [])
        original_count = len(blockers)
        self._data["blockers"] = [b for b in blockers if b.get("issue") != issue]
        removed = original_count - len(self._data["blockers"])

        if removed > 0:
            return self._persisted_result(
                confirmation=f"Blocker {issue} resolved and removed.",
                data={"resolved": True, "issue": issue, "removed_count": removed},
            )
        return ToolResult.failed(
            f"No blocker found with issue: {issue}",
            data={"issue": issue, "removed_count": 0},
        )

    # ------------------------------------------------------------------
    # Tools: GitHub-powered (delegated to sub-modules)
    # ------------------------------------------------------------------

    @tool(
        name="morning_signal",
        description="Generate a morning strategic briefing -- milestone status, blockers, recommended work items. Pulls live data from GitHub when GITHUB_TOKEN is available.",
        category=ToolCategory.SYSTEM,
        command_prefix="!morning",
    )
    async def morning_signal(self) -> ToolResult:
        """Generate the Morning Signal briefing from strategic memory + live GitHub data."""
        briefing = await generate_morning_signal(self._data)
        return ToolResult.ok(
            confirmation=briefing,
            data={"briefing": briefing},
        )

    def _dispatch_workflow_name(self) -> str:
        """Return the stable workflow capability selected by strategy config."""

        config = self._data.get("morning_signal_config", {})
        configured = (
            config.get("dispatch_workflow") if isinstance(config, dict) else None
        )
        name = str(configured or _DEFAULT_DISPATCH_WORKFLOW).strip()
        return name or _DEFAULT_DISPATCH_WORKFLOW

    def _resolve_dispatch_workflow(self, name: str):
        """Resolve one live contributed workflow without importing its owner."""

        registry = getattr(self.agent, "operator_registry", None)
        resolve = getattr(registry, "get_workflow_registration", None)
        if not callable(resolve):
            return None
        try:
            return resolve(name)
        except Exception:  # noqa: BLE001 - capability lookup fails closed
            logger.exception(
                "Strategic Memory could not inspect workflow capability %r", name
            )
            return None

    def _dispatch_session_id(self) -> str:
        """Use the live turn session when present, else a system audit scope."""

        resolve = getattr(self.agent, "get_turn_bound_session_id", None)
        if callable(resolve):
            try:
                session_id = resolve()
            except Exception:  # noqa: BLE001 - audit fallback is deterministic
                session_id = None
            if isinstance(session_id, str) and session_id.strip():
                return session_id.strip()
        return "strategic-memory"

    @staticmethod
    def _dispatch_runner_payload(result: Any) -> Dict[str, Any]:
        if isinstance(result, ToolResult):
            return result.to_dict()
        if isinstance(result, dict):
            return dict(result)
        return {"result": str(result)}

    @tool(
        name="signal_dispatch",
        description=(
            "Pick the highest-priority issue from strategic memory and start "
            "it through a live feature-contributed dispatch workflow. Preview "
            "with mode='suggest'; execute fails closed when no compatible "
            "workflow capability and governed runner are enabled."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!dispatch",
    )
    async def signal_dispatch(self, mode: str = "execute") -> ToolResult:
        """Preview or execute the top issue through generic workflow contracts."""

        mode = _normalize_choice(mode or "", _DISPATCH_MODE_ALIASES)
        if mode not in ("execute", "suggest"):
            return ToolResult.failed(
                f"Invalid mode '{mode}'. Must be one of: execute, suggest.",
                data={"mode": mode, "dispatched": False},
            )

        issue = await pick_top_issue(self._data)
        workflow_name = self._dispatch_workflow_name()
        if not issue:
            return ToolResult.ok(
                confirmation=(
                    "## Signal Dispatch"
                    + (" (suggest)" if mode == "suggest" else "")
                    + "\nNo actionable issue found."
                ),
                data={
                    "mode": mode,
                    "issue": None,
                    "workflow": workflow_name,
                    "dispatched": False,
                },
            )

        if mode == "suggest":
            body = (
                "## Signal Dispatch (suggest)\n"
                f"**Top issue:** {issue['repo']}#{issue['issue_number']}: "
                f"{issue['issue_title']}\n"
                f"**Priority:** {issue['priority']}\n"
                f"**Context:** {issue.get('context', 'N/A')}\n\n"
                f"Execute mode will request the contributed `{workflow_name}` "
                "workflow; no work was started by this preview."
            )
            return ToolResult.ok(
                confirmation=body,
                data={
                    "mode": "suggest",
                    "issue": issue,
                    "workflow": workflow_name,
                    "dispatched": False,
                    "body": body,
                },
            )

        registration = self._resolve_dispatch_workflow(workflow_name)
        if registration is None:
            return ToolResult.failed(
                f"Dispatch capability '{workflow_name}' is unavailable. Install "
                "and enable a feature that contributes that workflow, and enable "
                "a workflow runner exposing governed 'workflow_run'; the selected "
                "issue was not dispatched.",
                data={
                    "mode": "execute",
                    "issue": issue,
                    "workflow": workflow_name,
                    "dispatched": False,
                    "reason_code": "DISPATCH_CAPABILITY_UNAVAILABLE",
                },
            )

        execute_named_tool = getattr(self.agent, "execute_named_tool", None)
        if not callable(execute_named_tool):
            return ToolResult.failed(
                "Governed named-tool dispatch is unavailable on this agent "
                "runtime. Upgrade the host before enabling Strategic Memory "
                "dispatch; the selected issue was not dispatched.",
                data={
                    "mode": "execute",
                    "issue": issue,
                    "workflow": workflow_name,
                    "dispatched": False,
                    "reason_code": "GOVERNED_DISPATCH_UNAVAILABLE",
                },
            )

        params = {
            "repo": issue["repo"],
            "issue": issue["issue_number"],
            "issue_title": issue["issue_title"],
            "priority": issue["priority"],
            "context": issue.get("context", ""),
        }
        try:
            runner_result = await execute_named_tool(
                _WORKFLOW_RUN_TOOL,
                {"name": workflow_name, "params": params},
                session_id=self._dispatch_session_id(),
                source="strategic_memory.signal_dispatch",
            )
        except ToolNotRegisteredError:
            logger.info(
                "Strategic Memory dispatch workflow %r has no governed runner",
                workflow_name,
            )
            return ToolResult.failed(
                "The dispatch workflow is registered, but no enabled feature "
                "exposes the governed 'workflow_run' runner. Enable the "
                "Workflows feature and retry; the selected issue was not "
                "dispatched.",
                data={
                    "mode": "execute",
                    "issue": issue,
                    "workflow": workflow_name,
                    "dispatched": False,
                    "reason_code": "WORKFLOW_RUNNER_UNAVAILABLE",
                },
            )
        except Exception:  # noqa: BLE001 - sanitize the tool boundary
            logger.exception(
                "Strategic Memory dispatch runner failed for workflow %r",
                workflow_name,
            )
            return ToolResult.failed(
                "The governed workflow runner failed before accepting the "
                "selected issue. Inspect workflow diagnostics and retry; no "
                "dispatch was confirmed.",
                data={
                    "mode": "execute",
                    "issue": issue,
                    "workflow": workflow_name,
                    "dispatched": False,
                    "reason_code": "WORKFLOW_RUNNER_FAILED",
                },
            )

        runner_payload = self._dispatch_runner_payload(runner_result)
        accepted = (
            isinstance(runner_result, ToolResult)
            and runner_result.status.value == "ok"
        ) or (
            isinstance(runner_result, dict)
            and (
                runner_result.get("status") == "ok"
                or runner_result.get("success") is True
            )
        )
        if not accepted:
            runner_error = (
                runner_payload.get("error")
                or "the workflow runner did not confirm acceptance"
            )
            return ToolResult.failed(
                f"Dispatch workflow '{workflow_name}' did not start: "
                f"{runner_error}",
                data={
                    "mode": "execute",
                    "issue": issue,
                    "workflow": workflow_name,
                    "capability_owner": getattr(registration, "owner", ""),
                    "dispatched": False,
                    "reason_code": "WORKFLOW_RUN_REJECTED",
                    "runner_result": runner_payload,
                },
            )

        result_data = runner_payload.get("data")
        run_id = result_data.get("run_id") if isinstance(result_data, dict) else None
        body = (
            f"Dispatch workflow '{workflow_name}' accepted "
            f"{issue['repo']}#{issue['issue_number']}"
            + (f" as run {run_id}." if run_id else ".")
        )
        return ToolResult.ok(
            confirmation=body,
            data={
                "mode": "execute",
                "issue": issue,
                "workflow": workflow_name,
                "workflow_run_id": run_id,
                "capability_owner": getattr(registration, "owner", ""),
                "dispatched": True,
                "runner_result": runner_payload,
            },
        )

    @tool(
        name="backlog_hygiene",
        description="Scan all repos for backlog hygiene issues: missing assignees, milestones, status labels. Reports gaps and flags items needing human review.",
        category=ToolCategory.SYSTEM,
        command_prefix="!hygiene",
    )
    async def backlog_hygiene(self, fix: str = "no") -> ToolResult:
        """Scan repos for backlog hygiene issues and optionally auto-fix.

        Args:
            fix: Set to 'yes' to auto-fix issues where possible (add labels). Default 'no' (report only).
        """
        report = await run_backlog_hygiene(self._data, fix=fix)

        # Prerequisite failures (missing scan_repos / missing
        # GITHUB_TOKEN) come back as success-shaped strings from the
        # helper. Surface them as ERROR — the scan never ran, no fixes
        # could possibly have been applied, and a downstream caller
        # checking status must see failure rather than "applied=True".
        if _is_github_prereq_failure(report):
            return ToolResult.failed(
                report,
                data={"fix": fix, "report": report, "applied": False},
            )

        # Honesty: dry-run mode (anything but the runner's truthy
        # predicate) reports issues but does not change anything.
        # Surface as PARTIAL so the agent must speak that the report
        # is read-only — narrating "fixed N issues" off a dry run
        # would be a lie. Same pattern as model.cleanup_models(dry_run)
        # in PR #1098. Use the shared `is_auto_fix` predicate so this
        # wrapper agrees with the runner on what counts as auto-fix
        # (yes / true / 1, case-insensitive).
        if not is_auto_fix(fix):
            return ToolResult.partial(
                confirmation=report,
                error=(
                    f"fix={fix!r}: this is a report-only scan, no changes "
                    "were made. Re-run with fix='yes' (or 'true' / '1') "
                    "to apply auto-fixes."
                ),
                data={"fix": fix, "report": report, "applied": False},
            )
        return ToolResult.ok(
            confirmation=report,
            data={"fix": fix, "report": report, "applied": True},
        )

    @tool(
        name="session_log",
        description="End-of-day session log collector. Scans all repos for today's activity (issues closed, PRs merged, comments, commits) and generates a structured session summary with outcomes and metrics.",
        category=ToolCategory.SYSTEM,
        command_prefix="!sessionlog",
    )
    async def session_log(self, session_id: str = "", focus: str = "") -> ToolResult:
        """Collect end-of-day session log from GitHub activity.

        Args:
            session_id: Session number (e.g. '020'). Auto-generated if empty.
            focus: Brief description of today's focus area.
        """
        log = await collect_session_log(self._data, session_id=session_id, focus=focus)

        # Same shape as backlog_hygiene: prereq failures (no scan_repos
        # / no GITHUB_TOKEN) come back as text. The session log was not
        # collected; downstream callers branching on status must see
        # ERROR rather than treating the warning as a real log.
        if _is_github_prereq_failure(log):
            return ToolResult.failed(
                log,
                data={"session_id": session_id, "focus": focus, "log": log},
            )

        return ToolResult.ok(
            confirmation=log,
            data={"session_id": session_id, "focus": focus, "log": log},
        )

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
