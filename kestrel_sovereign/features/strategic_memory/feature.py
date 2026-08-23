"""
Strategic Memory Feature for Kestrel agents.

Provides persistent strategic context (vision, milestones, stakeholders,
decisions, blockers, patterns) that survives across sessions.

Two canonical files, split by how the content is meant to be read:

* ``STRATEGY.yaml`` -- the standing brief (vision, milestones, stakeholders,
  decisions). A bootstrap file: injected into the system prompt every turn by
  the BootstrapLoader, and therefore deliberately kept small.
* ``STRATEGY_LEDGER.yaml`` -- the growing logs (patterns learned, blockers).
  Not a bootstrap file. Reached by query through the graph projection in
  :mod:`ledger_index`, because a 175 KB append-only log poured into a 20,000
  character prompt budget does not inform an agent, it truncates at a byte
  offset (#2954).

This feature also provides !strategy commands for querying and updating
strategic context at runtime.
"""

import copy
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.enum_coerce import normalize_choice as _normalize_choice

from .backlog_hygiene import is_auto_fix, run_backlog_hygiene
from .blocker_reconcile import AMBIGUOUS_REPO, check_blockers, configured_repos
from .decision_index import decision_entries, project_decisions
from .issue_selection import pick_top_issue
from .ledger import (
    BLOCKERS_KEY,
    LEDGER_FILENAME,
    PATTERNS_KEY,
    StrategyLedger,
    active_blockers,
    active_patterns,
    has_ledger_sections,
    strip_ledger_sections,
)
from .ledger_index import (
    ACTIVE_STATUS,
    BLOCKER_SECTION,
    PATTERN_SECTION,
    LedgerSection,
    index_membership,
    project_ledger,
    recall_nodes,
    search_rows,
)
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


#: How many patterns ``strategy_view patterns`` renders before pointing at the
#: query layer. The whole point of moving the log out of the prompt was to stop
#: pouring hundreds of rows into a context window; rendering all of them here
#: would rebuild the same problem inside a tool result. The renderer says how
#: many it withheld — a silently capped list is a lie about the record.
_PATTERN_VIEW_LIMIT = 25

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

    # ``patterns`` is deliberately absent. The template used to ship it empty
    # while every reader and writer used ``patterns_learned``, so agents ended
    # up carrying both keys -- one empty, one holding hundreds of rows. Two
    # names for one concept is one name too many; the ledger owns it now.
    _DEFAULT_TEMPLATE = {
        "version": 1,
        "vision": "Define your agent's long-term vision here.",
        "milestones": [],
        "stakeholders": [],
        "decisions": [],
    }

    def __init__(self, agent):
        super().__init__(agent)
        self._data: Dict[str, Any] = {}
        self._strategy_path: Optional[Path] = None
        self._ledger = StrategyLedger(None)

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
                    # Create default template so the agent can start capturing
                    # strategy. deepcopy, not dict(): a shallow copy shares the
                    # class-level ``milestones``/``stakeholders``/``decisions``
                    # LIST OBJECTS with every other instance in the process, so
                    # ``strategy_add_decision`` appended into the template
                    # itself. On a multi-agent host the next agent created was
                    # then born holding the previous agent's decisions, and
                    # ``_save()`` wrote them into its brand-new STRATEGY.yaml --
                    # one agent's strategic record leaking into another's, on
                    # disk. Noticed while adding the index consumers (#2954);
                    # the bug predates them.
                    self._data = copy.deepcopy(self._DEFAULT_TEMPLATE)
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

        # The ledger is loaded even when STRATEGY.yaml failed above: the two
        # files are independent canonical records, and a broken brief must not
        # take the pattern/blocker log down with it.
        try:
            self._ledger = StrategyLedger(
                self._strategy_path.parent / LEDGER_FILENAME
                if self._strategy_path
                else None
            )
            self._ledger.load()
            self._migrate_ledger_sections()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load strategy ledger: {e}")

        # Rebuild the graph index from what YAML says (#2851). Doing it at load
        # is what makes the index genuinely derived: an agent whose database was
        # lost, or whose STRATEGY.yaml was edited by hand or pulled from git,
        # gets a correct index on next start with no migration step. It is also
        # how decisions recorded before this existed become reachable at all.
        await self._reindex_decisions()
        await self._reindex_ledger()

    # ------------------------------------------------------------------
    # Ledger: migration, persistence, projection
    # ------------------------------------------------------------------

    def _migrate_ledger_sections(self) -> Dict[str, Any]:
        """Move ``patterns_learned``/``blockers`` out of STRATEGY.yaml.

        Ordered so that no ordering can lose a row: the ledger is written
        first, and STRATEGY.yaml only gives the sections up once that write is
        confirmed persisted. An interrupted migration therefore leaves the rows
        duplicated across both files -- recoverable, and converged by the next
        run, because :meth:`StrategyLedger.absorb` skips ids it already holds.
        """
        report = {"migrated": False, "patterns": 0, "blockers": 0}
        if not self._ledger.readable:
            # The file exists and could not be understood. Migrating into it
            # would write over contents we never read.
            report["error"] = self._ledger.load_error
            return report

        if not has_ledger_sections(self._data):
            # Already migrated (or a fresh agent). Still normalize, so
            # hand-written rows without ids get addressable -- and persist
            # that, because an id minted only in memory is an address that
            # changes on the next restart, orphaning every graph node and
            # every id the agent wrote down.
            if self._ledger.needs_save and self._ledger.active:
                error = self._ledger.save()
                if error:
                    logger.warning(
                        "Strategy ledger ids minted in memory but not "
                        "persisted: %s",
                        error,
                    )
                    report["error"] = error
            return report

        absorbed = self._ledger.absorb(self._data)
        if absorbed.get("error"):
            report["error"] = absorbed["error"]
            return report
        self._ledger.normalize()
        error = self._ledger.save()
        if error:
            logger.error(
                "Strategy ledger migration deferred -- %s. STRATEGY.yaml is "
                "unchanged and still holds the rows.",
                error,
            )
            report["error"] = error
            return report

        if strip_ledger_sections(self._data):
            outcome = self._save()
            if not outcome.persisted:
                # The ledger already has the rows; STRATEGY.yaml keeping its
                # copy for now is a duplicate, not a loss. Re-read it so the
                # in-memory brief matches the file that is actually on disk.
                logger.warning(
                    "Strategy ledger written but STRATEGY.yaml could not be "
                    "trimmed: %s",
                    outcome.error,
                )
                report["error"] = outcome.error
        report["migrated"] = True
        report["patterns"] = absorbed["patterns"]
        report["blockers"] = absorbed["blockers"]
        if absorbed["patterns"] or absorbed["blockers"]:
            logger.info(
                "Strategy ledger migration: %d pattern(s), %d blocker(s) moved "
                "to %s",
                absorbed["patterns"],
                absorbed["blockers"],
                LEDGER_FILENAME,
            )
        return report

    def _strategy_data_view(self) -> Dict[str, Any]:
        """STRATEGY.yaml plus the ledger's *active* rows, for readers.

        The split is a persistence decision, not a behaviour change:
        ``morning_signal``, ``pick_top_issue`` and the section renderers asked
        for ``data["blockers"]`` before and must still get blockers. They see
        the active rows only -- a resolved blocker is history, not a blocker.

        Deliberately a copy. Merging the ledger into ``self._data`` would put
        the ledger's rows back into whatever ``_save()`` writes to
        STRATEGY.yaml, undoing the migration on the next write.
        """
        view = dict(self._data)
        view[BLOCKERS_KEY] = active_blockers(self._ledger.blockers)
        view[PATTERNS_KEY] = active_patterns(self._ledger.patterns)
        return view

    def _resolve_blocker_repo(
        self, issue: str, declared: str
    ) -> tuple[str, Optional[str]]:
        """Decide which repository a new blocker's issue belongs to.

        Returns ``(repo, error)``. Three unambiguous sources, in order: what
        the caller declared, a fully qualified ``owner/repo#N`` issue, and a
        lone configured ``scan_repos`` entry. When several repositories are
        configured and none of those apply, this refuses rather than guessing
        -- the guess is what made reconciliation resolve a blocker against the
        wrong project's issue 42.

        An unqualified issue with *no* configured repos is left unbound: there
        is nothing to be ambiguous between, and refusing would block recording
        a blocker on a host with no GitHub configuration at all.
        """
        declared = str(declared or "").strip()
        if declared:
            return declared, None
        raw = str(issue or "").strip()
        if "#" in raw and "/" in raw.split("#", 1)[0]:
            return raw.split("#", 1)[0], None
        configured = configured_repos(self._data)
        if len(configured) == 1:
            return configured[0], None
        if len(configured) > 1 and str(issue or "").strip():
            return "", (
                f"{len(configured)} repositories are configured "
                f"({', '.join(configured)}), so issue {issue!r} is ambiguous. "
                "Pass repo='owner/repo', or write the issue as "
                "'owner/repo#123'."
            )
        return "", None

    def _ledger_unreadable_result(self, data: Dict[str, Any]) -> ToolResult:
        """The refusal every ledger tool returns when the file is unreadable.

        Fail closed, and say so. The alternative -- carrying on with an empty
        in-memory ledger -- reads as "you have no patterns" and, on the first
        write, becomes true.
        """
        return ToolResult.failed(
            f"Strategy ledger is unavailable: {self._ledger.load_error}. "
            "Nothing was read or written; repair or move the file and restart.",
            data={**data, "ledger_unavailable": True},
        )

    def _persisted_ledger_result(
        self, confirmation: str, data: Dict[str, Any]
    ) -> ToolResult:
        """Turn a ledger write into an honest ToolResult, as ``_save`` does."""
        if not self._ledger.readable:
            return self._ledger_unreadable_result(
                {**data, "persisted": False}
            )
        if not self._ledger.path:
            return ToolResult.failed(
                "No strategy ledger path configured -- strategic memory is "
                "not active, so nothing was persisted.",
                data={**data, "persisted": False},
            )
        error = self._ledger.save()
        if error:
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"In-memory update applied but the write failed: {error}"
                ),
                data={**data, "persisted": False},
            )
        return ToolResult.ok(
            confirmation=confirmation,
            data={**data, "persisted": True},
        )

    async def _reindex_ledger(self) -> Dict[str, Any]:
        """Project the ledger into the graph.

        Never raises and never touches the ledger file: the canonical record is
        already on disk, so a graph failure must not turn into a strategic-
        memory failure. Reconciles rather than merely upserting, for the same
        reason the decision index does -- a row deleted from the canonical file
        must stop being reachable through the index.
        """
        if not self._ledger.readable:
            # An unreadable ledger is not an empty one. Reconciliation derives
            # its keep-set from the ledger's contents, so projecting a failed
            # parse would present "no rows" as "every row was deleted" and take
            # the whole derived index with it -- a parse error escalated into
            # data loss. Exactly the failure the decision index hit in #2851
            # when its keep-set came from a failed read rather than from
            # canonical membership.
            logger.warning(
                "strategy ledger index skipped: ledger unreadable (%s)",
                self._ledger.load_error,
            )
            return {"projected": 0, "skipped": 0, "failed": 0,
                    "skipped_reason": "ledger_unavailable"}
        agent_id = self._projection_agent_id()
        if not agent_id:
            logger.warning(
                "strategy ledger index skipped: agent exposes no agent_id/did/id"
            )
            return {"projected": 0, "skipped": 0, "failed": 0,
                    "skipped_reason": "no_agent_identity"}
        try:
            storage = getattr(self.agent, "storage", None)
            graph_store = getattr(storage, "graph", None) if storage else None
            report = await project_ledger(graph_store, agent_id, self._ledger)
        except Exception as e:  # noqa: BLE001 - the index is best-effort
            logger.warning("strategy ledger index projection failed: %s", e)
            return {"projected": 0, "skipped": 0, "failed": 0, "error": str(e)}
        if report.get("failed") or report.get("skipped_reason"):
            logger.info("strategy ledger index: %s", report)
        return report

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
        if not self._data and not (self._ledger.patterns or self._ledger.blockers):
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
        if not self._ledger.readable and section in ("blockers", "patterns"):
            # These two sections are the ledger, entirely. Rendering them from
            # an unread file would print an empty list under a heading that
            # claims to be the record.
            return self._ledger_unreadable_result({"section": section})
        body = renderer()
        if not self._ledger.readable and section == "all":
            # ``all`` still has a genuine brief to show -- vision, milestones,
            # stakeholders and decisions come from STRATEGY.yaml, which is a
            # separate file and is fine. Refusing the whole view would let a
            # broken ledger take down a working brief, the mirror of the rule
            # that a broken brief must not take down the ledger. So it renders,
            # and says plainly which part of it is missing rather than showing
            # an empty Blockers heading and letting the reader assume zero.
            #
            # Only ``all``: the single-section views above draw nothing from
            # the ledger, so caveating ``!strategy vision`` with "patterns and
            # blockers are missing" would report a loss that view never had.
            return ToolResult.partial(
                confirmation=body,
                error=(
                    f"Patterns and blockers are missing from this view: "
                    f"{self._ledger.load_error}. Every other section is "
                    "complete."
                ),
                data={
                    "section": section,
                    "body": body,
                    "ledger_unavailable": True,
                },
            )
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
        self,
        issue: str,
        title: str,
        severity: str = "medium",
        owner: str = "unassigned",
        repo: str = "",
        notes: str = "",
    ) -> ToolResult:
        """
        Add a blocker to the strategic memory.

        Args:
            issue: Issue number or identifier
            title: Short description of the blocker
            severity: How severe -- one of low, medium, high, critical (default medium)
            owner: Who owns resolving this blocker
            repo: owner/repo the issue lives in. Inferred from a qualified issue or a single configured scan repo; required when several are configured.
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
        if not self._ledger.readable:
            return self._ledger_unreadable_result({"recorded": False})

        # Bind the repository now, while the caller is here to say which one.
        # A row written without it is a row a later reconcile has to guess
        # about, and "#42 is closed" is only true of a specific repository --
        # binding to the first configured repo that happens to have an issue 42
        # is how a blocker gets resolved against a different project's ticket.
        resolved_repo, repo_error = self._resolve_blocker_repo(issue, repo)
        if repo_error:
            return ToolResult.failed(
                repo_error, data={"recorded": False, "issue": issue}
            )
        entry = self._ledger.add_blocker(
            issue=issue,
            title=title,
            severity=severity,
            owner=owner,
            notes=notes,
            repo=resolved_repo,
        )
        result = self._persisted_ledger_result(
            confirmation=f"Blocker recorded: {title} (id {entry['id']})",
            data={"recorded": True, "blocker": entry, "blocker_id": entry["id"]},
        )
        # Index only what reached the canonical file, for the same reason
        # ``strategy_add_decision`` does: a derived index must not publish a row
        # its source does not contain.
        if (result.data or {}).get("persisted"):
            await self._reindex_ledger()
        return result

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
        if not self._ledger.readable:
            # Refuse before mutating, not after. ``_persisted_ledger_result``
            # would catch the write, but the row would already be sitting in
            # the in-memory ledger, where ``strategy_search`` and
            # ``strategy_view`` would show it as recorded -- a row that exists
            # nowhere on disk and vanishes on restart.
            return self._ledger_unreadable_result({"recorded": False})
        entry = self._ledger.add_pattern(
            pattern=pattern, source=source, implication=implication
        )
        result = self._persisted_ledger_result(
            confirmation=f"Pattern recorded: {pattern} (id {entry['id']})",
            data={"recorded": True, "pattern": entry, "pattern_id": entry["id"]},
        )
        if (result.data or {}).get("persisted"):
            await self._reindex_ledger()
        return result

    @tool(
        name="strategy_supersede_pattern",
        description=(
            "Retire a learned pattern by its id, optionally naming the pattern "
            "that replaces it. The row is kept as history, not deleted."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def strategy_supersede_pattern(
        self, pattern_id: str, reason: str = "", superseded_by: str = ""
    ) -> ToolResult:
        """
        Mark a learned pattern superseded so it drops out of the active set.

        Args:
            pattern_id: The pattern row id (from strategy_add_pattern / strategy_search)
            reason: Why the pattern no longer holds
            superseded_by: Optional id of the pattern that replaces it
        """
        if not self._ledger.readable:
            # Without this the lookup below runs against the empty default
            # ledger and returns "No pattern found with id" -- a false negative
            # that reads as "you never recorded that", when the truth is that
            # the file holding it could not be read.
            return self._ledger_unreadable_result(
                {"pattern_id": pattern_id, "superseded": False}
            )
        key, row = self._ledger.find(pattern_id)
        if row is None or key != PATTERNS_KEY:
            return ToolResult.failed(
                f"No pattern found with id: {pattern_id}. Use strategy_search "
                "to find a pattern's id.",
                data={"pattern_id": pattern_id, "superseded": False},
            )
        if row.get("superseded_at"):
            return ToolResult.failed(
                f"Pattern {pattern_id} was already superseded on "
                f"{row['superseded_at']}.",
                data={"pattern_id": pattern_id, "superseded": False},
            )
        if superseded_by:
            replacement_key, replacement = self._ledger.find(superseded_by)
            if replacement is None or replacement_key != PATTERNS_KEY:
                return ToolResult.failed(
                    f"No pattern found with id: {superseded_by} -- refusing to "
                    "record a replacement that does not exist.",
                    data={"pattern_id": pattern_id, "superseded": False},
                )

        self._ledger.supersede_pattern(
            row, reason=reason, superseded_by=superseded_by
        )
        result = self._persisted_ledger_result(
            confirmation=f"Pattern {pattern_id} superseded.",
            data={"superseded": True, "pattern_id": pattern_id, "pattern": row},
        )
        if (result.data or {}).get("persisted"):
            await self._reindex_ledger()
        return result

    @tool(
        name="strategy_resolve_blocker",
        description=(
            "Resolve one blocker by its row id (preferred) or by a unique issue "
            "identifier. Refuses an issue key that matches several rows -- "
            "resolve those individually by id."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def strategy_resolve_blocker(
        self, issue: str, resolution: str = ""
    ) -> ToolResult:
        """
        Resolve a blocker by its row id or issue identifier.

        Args:
            issue: The blocker row id, or the issue number/identifier
            resolution: Optional note describing how it was resolved
        """
        if not self._ledger.readable:
            # Same false negative as strategy_supersede_pattern: an empty
            # in-memory ledger answers "no blocker found with issue X" for a
            # blocker that is sitting in the file, unread.
            return self._ledger_unreadable_result(
                {"issue": issue, "removed_count": 0}
            )
        matches = self._ledger.blockers_matching(issue)
        if not matches:
            return ToolResult.failed(
                f"No blocker found with issue: {issue}",
                data={"issue": issue, "removed_count": 0},
            )
        if len(matches) > 1:
            # The defect this ticket was filed on: matching by issue key alone
            # removed every row that shared it -- one call returned
            # removed_count 10. Ambiguity is now a refusal with the ids needed
            # to be specific, not a bulk delete the caller never asked for.
            ids = [str(m.get("id") or "?") for m in matches]
            return ToolResult.failed(
                f"{len(matches)} blockers share issue {issue}: "
                + ", ".join(ids)
                + ". Resolve them one at a time by id.",
                data={
                    "issue": issue,
                    "removed_count": 0,
                    "ambiguous": True,
                    "candidate_ids": ids,
                },
            )

        row = matches[0]
        self._ledger.resolve_blocker(row, resolution=resolution)
        result = self._persisted_ledger_result(
            confirmation=f"Blocker {row.get('id')} ({issue}) resolved.",
            data={
                "resolved": True,
                "issue": issue,
                "blocker_id": row.get("id"),
                "removed_count": 1,
            },
        )
        if (result.data or {}).get("persisted"):
            await self._reindex_ledger()
        return result

    @tool(
        name="strategy_search",
        description=(
            "Search the strategy ledger (learned patterns and blockers) by "
            "keyword. This is the query path that replaced dumping the whole "
            "log into the system prompt."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!strategy-search",
    )
    async def strategy_search(
        self,
        query: str,
        kind: str = "all",
        limit: int = 10,
        include_retired: bool = False,
    ) -> ToolResult:
        """
        Search learned patterns and blockers.

        Args:
            query: Words to match against the ledger rows
            kind: Which rows to search -- all, patterns, blockers (default all)
            limit: Maximum matches to return (default 10)
            include_retired: Include superseded patterns and resolved blockers
        """
        # search_rows has always taken this; not exposing it made this tool
        # unable to answer for retired rows, while recall_patterns/
        # recall_blockers named it as the canonical fallback when their
        # retired-mode index came back stale (#3064).
        if not isinstance(include_retired, bool):
            return ToolResult.failed(
                "include_retired must be a boolean, got "
                f"{type(include_retired).__name__}={include_retired!r}"
            )
        kind = (kind or "all").strip().lower()
        if kind not in ("all", "patterns", "pattern", "blockers", "blocker"):
            return ToolResult.failed(
                f"Unknown kind: {kind}. Available: all, patterns, blockers.",
                data={"kind": kind},
            )
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"limit must be an integer, got {limit!r}", data={"limit": limit}
            )
        if not str(query or "").strip():
            return ToolResult.failed(
                "strategy_search needs a non-empty query.", data={"query": query}
            )
        if not self._ledger.readable:
            # "No ledger matches" from an unread ledger is the exact shape of
            # the defect this ticket was filed on: a truthful-looking zero for
            # a question whose real answer is non-empty.
            return self._ledger_unreadable_result(
                {"query": query, "kind": kind, "matches": [], "count": 0}
            )

        matches = search_rows(
            self._ledger.data,
            query,
            kind=kind,
            limit=limit,
            include_retired=include_retired,
        )
        if not matches:
            return ToolResult.ok(
                confirmation=f"No ledger matches for {query!r}.",
                data={"query": query, "kind": kind, "matches": [], "count": 0},
            )
        lines = [f"## Ledger matches for {query!r}"]
        for match in matches:
            row = match["row"]
            text = row.get("pattern") if match["kind"] == "pattern" else row.get("title")
            lines.append(f"- [{match['kind']} {match['id']}] {text}")
            detail = (
                row.get("implication")
                if match["kind"] == "pattern"
                else row.get("notes")
            )
            if detail:
                lines.append(f"  {detail}")
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "query": query,
                "kind": kind,
                "count": len(matches),
                "matches": [
                    {
                        "id": m["id"],
                        "kind": m["kind"],
                        "score": m["score"],
                        "row": m["row"],
                    }
                    for m in matches
                ],
            },
        )

    # ------------------------------------------------------------------
    # Tools: reading the projected index back out of the graph
    # ------------------------------------------------------------------

    async def _recall_ledger(
        self,
        section: LedgerSection,
        limit: Any,
        include_retired: bool,
    ) -> ToolResult:
        """Shared body for :meth:`recall_patterns` / :meth:`recall_blockers`.

        Reads the GRAPH, deliberately, rather than the ledger this feature
        already holds in memory. A derived index nothing queries is a write
        nobody checked: #2851 shipped a decision projection whose only
        consumer was itself, and the gap was invisible because every reader
        went to YAML. Routing these two tools through the index means a
        projection that stops working fails a query instead of going quiet.
        """
        noun = section.noun
        if not isinstance(include_retired, bool):
            return ToolResult.failed(
                f"{section.include_flag_name} must be a boolean, got "
                f"{type(include_retired).__name__}={include_retired!r}"
            )
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1 or limit_val > 200:
            return ToolResult.failed(
                f"limit must be between 1 and 200, got {limit_val}"
            )

        agent_id = self._projection_agent_id()
        if not agent_id:
            return ToolResult.failed(
                "Agent exposes no agent_id/did/id, so the strategy index "
                "cannot be scoped to this agent -- refusing rather than "
                "returning another agent's rows or a misleading empty list."
            )
        storage = getattr(self.agent, "storage", None)
        graph_store = getattr(storage, "graph", None) if storage else None
        try:
            rows = await recall_nodes(
                graph_store,
                agent_id,
                section.node_type,
                include_retired=include_retired,
                limit=limit_val,
            )
        except Exception as e:  # noqa: BLE001
            # A query failure is not an empty result. Reporting zero here is
            # the precise lie this ticket was filed on.
            logger.error(
                "strategy index recall failed for %s: %s", section.node_type, e
            )
            return ToolResult.failed(
                f"Could not read the strategy index: {e}",
                data={"node_type": section.node_type, "count": 0},
            )

        data: Dict[str, Any] = {
            "count": len(rows),
            "limit_requested": limit_val,
            section.include_flag_name: include_retired,
            noun: rows,
        }
        confirmation = (
            f"Retrieved {len(rows)} {noun[:-1] if len(rows) == 1 else noun} "
            f"from the strategy index (limit requested: {limit_val})"
        )

        # The divergence check runs on EVERY result, not only the empty one.
        # #3064: a non-empty answer was taken as proof the index was complete,
        # so one projected row standing in for two canonical ones returned a
        # clean, short list. Non-emptiness answers a narrower question than
        # the caller asked.
        if not self._ledger.readable:
            # An unreadable ledger gives no trustworthy membership to compare
            # against, so the check has nothing to say -- and says so rather
            # than letting silence read as agreement.
            data["completeness_checked"] = False
            data["completeness_unchecked_reason"] = "ledger_unreadable"
            return ToolResult.ok(confirmation=confirmation, data=data)

        # Membership, not a count: an index holding one row the ledger dropped
        # and missing one it still holds has the right total and the wrong
        # contents. Read from the WHOLE scoped projection rather than from the
        # page, because a divergence that sorts behind every canonical row is
        # invisible to a LIMIT and present in the database -- the page would
        # certify a clean index and a larger limit would then return deleted
        # guidance (#3064).
        try:
            membership, membership_complete = await index_membership(
                graph_store, agent_id, section.node_type
            )
        except Exception as e:  # noqa: BLE001
            # The rows above are a real answer; only the check failed. Saying
            # nothing here would let a silent failure read as agreement.
            logger.debug(
                "strategy index membership unavailable for %s: %s",
                section.node_type,
                e,
            )
            data["completeness_checked"] = False
            data["completeness_unchecked_reason"] = "index_membership_unavailable"
            return ToolResult.ok(confirmation=confirmation, data=data)

        if not membership_complete:
            data["completeness_checked"] = False
            data["completeness_unchecked_reason"] = "index_exceeds_membership_cap"
            return ToolResult.ok(confirmation=confirmation, data=data)

        ledger_data = self._ledger.data
        # Membership is status-agnostic: whether a row is retired decides which
        # PAGE it belongs on, not whether the index should hold a node for it.
        all_ids = section.expected_row_id_list(ledger_data, include_retired=True)
        ledger_all = set(all_ids)
        ledger_active = section.expected_row_ids(ledger_data, include_retired=False)
        indexed = set(membership)
        # Rows, not ids: two canonical rows sharing an id are two rows, and
        # counting the set would under-report the ledger to the caller.
        data["canonical_expected"] = len(
            section.expected_rows(ledger_data, include_retired=include_retired)
        )
        data["completeness_checked"] = True

        missing = ledger_all - indexed
        orphaned = indexed - ledger_all
        # Two canonical rows on one id project to one node, so the second
        # overwrites the first and one row is unreachable however healthy the
        # projection is. Counted separately because reprojecting cannot fix it.
        colliding = len(all_ids) - len(ledger_all)
        # Membership is a strictly weaker question than agreement: an
        # id-stable edit leaves the id and the status identical while the
        # indexed row goes stale. What a healthy index would hold is asked of
        # the projection's own carry-over rule rather than reconstructed here.
        canonical_rows = {
            section.row_id(row): row
            for row in section.expected_rows(ledger_data, include_retired=True)
        }
        misfiled = set()
        drifted = set()
        for row_id, indexed in membership.items():
            row = canonical_rows.get(row_id)
            if row is None:
                continue  # an orphan, already counted
            expected = section.expected_properties(agent_id, row, indexed.properties)
            if str(expected.get("status") or "") != indexed.status:
                misfiled.add(row_id)
            elif expected != indexed.properties:
                drifted.add(row_id)

        if not (missing or orphaned or misfiled or colliding or drifted):
            return ToolResult.ok(confirmation=confirmation, data=data)

        data["index_stale"] = True
        divergences = []
        if missing:
            data["missing_count"] = len(missing)
            divergences.append(
                f"{len(missing)} of the {len(ledger_all)} {noun} "
                f"{LEDGER_FILENAME} holds are absent from it"
            )
        if orphaned:
            data["orphaned_count"] = len(orphaned)
            divergences.append(
                f"it holds {len(orphaned)} row(s) {LEDGER_FILENAME} no longer "
                "does"
            )
        if misfiled:
            data["misfiled_count"] = len(misfiled)
            divergences.append(
                f"{len(misfiled)} row(s) whose current/retired state it "
                f"disagrees with {LEDGER_FILENAME} on"
            )
        if drifted:
            data["drifted_count"] = len(drifted)
            divergences.append(
                f"{len(drifted)} row(s) whose indexed content no longer "
                f"matches {LEDGER_FILENAME}"
            )
        if colliding:
            data["colliding_count"] = colliding
            divergences.append(
                f"{colliding} canonical row(s) share an id with another, so "
                "the index can hold only one of each -- reprojecting cannot "
                f"fix that, {LEDGER_FILENAME} needs the duplicate ids resolved"
            )
        fallback = "strategy_search"
        if include_retired:
            # The advice has to name a path that can actually return the rows
            # it is about: strategy_search excludes retired rows unless told
            # otherwise, so recommending it bare for a retired-mode recall
            # promises a fallback that structurally cannot answer (#3064).
            fallback = "strategy_search(..., include_retired=True)"
        if missing or orphaned or misfiled or drifted:
            remedy = (
                "The index is stale or was never built -- restart the agent "
                f"to reproject, and use {fallback} to read the canonical file "
                "meanwhile."
            )
        else:
            # Collisions are the ONLY divergence here, and a restart would
            # reproduce the same overwrite. Telling the operator to reproject
            # after saying reprojection cannot fix it is advice that
            # contradicts itself.
            remedy = (
                f"Resolve the duplicate ids in {LEDGER_FILENAME}; until then "
                f"use {fallback} to read the canonical rows."
            )
        return ToolResult.partial(
            confirmation=confirmation,
            error=(
                f"The strategy index diverges from {LEDGER_FILENAME}: "
                + ", and ".join(divergences)
                + ". "
                + remedy
            ),
            data=data,
        )

    @tool(
        name="recall_patterns",
        description=(
            "Recall learned patterns from the strategy index (graph nodes of "
            "type 'strategy_pattern'). Superseded patterns are excluded by "
            "default; pass include_superseded=True to see them."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!patterns",
    )
    async def recall_patterns(
        self, limit: int = 25, include_superseded: bool = False
    ) -> ToolResult:
        """
        Recall learned patterns through the graph index.

        Args:
            limit: Maximum patterns to return (1-200, default 25)
            include_superseded: Include patterns that have been retired
        """
        return await self._recall_ledger(
            section=PATTERN_SECTION,
            limit=limit,
            include_retired=include_superseded,
        )

    @tool(
        name="recall_blockers",
        description=(
            "Recall blockers from the strategy index (graph nodes of type "
            "'strategy_blocker'). Resolved blockers are excluded by default; "
            "pass include_resolved=True to see them."
        ),
        category=ToolCategory.MEMORY,
        command_prefix="!blockers",
    )
    async def recall_blockers(
        self, limit: int = 25, include_resolved: bool = False
    ) -> ToolResult:
        """
        Recall blockers through the graph index.

        Args:
            limit: Maximum blockers to return (1-200, default 25)
            include_resolved: Include blockers that have been resolved
        """
        return await self._recall_ledger(
            section=BLOCKER_SECTION,
            limit=limit,
            include_retired=include_resolved,
        )

    @tool(
        name="strategy_reconcile_blockers",
        description=(
            "Check each active blocker against live GitHub issue state and "
            "report which reference already-closed issues. Pass apply='yes' to "
            "resolve the stale rows."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!strategy-reconcile",
    )
    async def strategy_reconcile_blockers(self, apply: str = "no") -> ToolResult:
        """
        Reconcile the blocker ledger against live GitHub state.

        Args:
            apply: Set to 'yes' to resolve blockers whose issue is closed. Default 'no' (report only).
        """
        if not self._ledger.readable:
            # An unread ledger has no active blockers, so the reconcile would
            # report "no active blockers to check" -- a clean bill of health
            # for a check that never saw the rows.
            return self._ledger_unreadable_result(
                {"applied": False, "checked": 0, "closed_count": 0}
            )
        report = await check_blockers(self._ledger.data, self._data)
        if not report.get("ran"):
            # The check never ran. Reporting "0 stale blockers" here would be
            # the same shape of lie the ticket was filed about.
            return ToolResult.failed(
                report.get("reason", "Blocker reconciliation could not run."),
                data={"applied": False, "report": report},
            )

        closed = report.get("closed", [])
        body = self._format_reconcile_report(report)
        data = {
            "applied": False,
            "checked": report.get("checked", 0),
            "closed_count": len(closed),
            "report": report,
        }
        if not is_auto_fix(apply):
            if not closed:
                return ToolResult.ok(confirmation=body, data=data)
            return ToolResult.partial(
                confirmation=body,
                error=(
                    f"apply={apply!r}: this is a report-only reconcile, no "
                    f"blocker was resolved. Re-run with apply='yes' to resolve "
                    f"the {len(closed)} stale row(s)."
                ),
                data=data,
            )

        resolved: List[str] = []
        for entry in closed:
            _, row = self._ledger.find(str(entry.get("id") or ""))
            if row is None:
                continue
            self._ledger.resolve_blocker(
                row,
                resolution=(
                    f"GitHub issue {entry.get('issue')} is closed "
                    f"({entry.get('repo') or 'unknown repo'})"
                ),
            )
            resolved.append(str(row.get("id")))
        data["resolved_ids"] = resolved
        if not resolved:
            # Nothing was written, so there is no persist outcome to report.
            # Claiming ``persisted: True`` off a run that touched no row would
            # be a small lie in the same envelope the honesty layer reads.
            return ToolResult.ok(
                confirmation=f"{body}\n\nNo blocker needed resolving.",
                data={**data, "applied": True},
            )
        result = self._persisted_ledger_result(
            confirmation=f"{body}\n\nResolved {len(resolved)} stale blocker(s).",
            data={**data, "applied": True},
        )
        if (result.data or {}).get("persisted"):
            await self._reindex_ledger()
        return result

    @staticmethod
    def _format_reconcile_report(report: Dict[str, Any]) -> str:
        lines = ["## Blocker Reconciliation"]
        checked = report.get("checked", 0)
        unchecked = report.get("unresolvable", [])
        if not checked and not unchecked:
            # "Every active blocker still references an open issue" would read
            # as a clean bill of health for a check that had nothing to check.
            return "\n".join(lines + ["No active blockers to check."])
        lines.append(f"Checked {checked} active blocker(s).")
        if unchecked:
            # Say it up front, not only in the section below: a run where every
            # row was skipped must not read like a run where every row passed.
            lines.append(
                f"{len(unchecked)} active blocker(s) could not be checked."
            )
        closed = report.get("closed", [])
        if closed:
            lines.append("")
            lines.append("### Referencing closed issues")
            for entry in closed:
                lines.append(
                    f"- [{entry.get('id')}] {entry.get('issue')}: "
                    f"{entry.get('title')}"
                )
        if unchecked:
            lines.append("")
            lines.append("### Could not be checked")
            for entry in unchecked:
                line = f"- [{entry.get('id')}] {entry.get('issue')}"
                if entry.get("reason") == AMBIGUOUS_REPO:
                    # Name the actual problem. "Could not be checked" alone
                    # invites the reader to assume a transient GitHub failure,
                    # when the fix is one field on the row.
                    candidates = ", ".join(entry.get("candidate_repos") or [])
                    line += (
                        f" -- names no repository, and {candidates} are all "
                        "configured. Set repo on the row to check it."
                    )
                lines.append(line)
        if not closed and not unchecked:
            lines.append("Every active blocker still references an open issue.")
        return "\n".join(lines)

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
        # The merged view, not ``self._data``: the briefing has always led with
        # blockers, and the file split must not quietly empty that section.
        briefing = await generate_morning_signal(self._strategy_data_view())
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

        issue = await pick_top_issue(self._strategy_data_view())
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
        blockers = self._strategy_data_view().get(BLOCKERS_KEY, [])
        if not blockers:
            return "## Blockers\nNo active blockers."
        lines = ["## Blockers"]
        for b in blockers:
            lines.append(
                f"- [{str(b.get('severity', '?')).upper()}] "
                f"{b.get('issue', '?')}: {b.get('title', '?')} "
                f"(id: {b.get('id', '?')}, owner: {b.get('owner', 'unassigned')})"
            )
        return "\n".join(lines)

    def _format_patterns(self) -> str:
        patterns = self._strategy_data_view().get(PATTERNS_KEY, [])
        if not patterns:
            return "## Patterns Learned\nNone recorded."
        lines = ["## Patterns Learned"]
        for p in patterns[:_PATTERN_VIEW_LIMIT]:
            lines.append(f"- [{p.get('id', '?')}] {p.get('pattern', '?')}")
            impl = p.get("implication")
            if impl:
                lines.append(f"  Implication: {impl}")
        withheld = len(patterns) - _PATTERN_VIEW_LIMIT
        if withheld > 0:
            # Say what was withheld. A capped list that reads like a complete
            # one is the same defect as a prompt truncated at a byte offset,
            # only quieter.
            lines.append(
                f"\n({withheld} more active pattern(s) not shown -- use "
                f"strategy_search to reach them.)"
            )
        return "\n".join(lines)
