"""TalonCoordinatorFeature - lightweight dispatch to external Talon daemon.

Wraps coordination, not the engine. Dispatches work to the external
kestrel-talon daemon via Agent Mesh Protocol (preferred) or CLI fallback.

Reference: sovereign #301
"""

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign._async_process import (
    SubprocessCleanupError,
    start_async_process,
    terminate_process_tree,
)
from kestrel_sovereign._bounded_subprocess import run_bounded_subprocess
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.cli.terminal import redact_secrets
from kestrel_sovereign.features.talon.wait_provider import TalonWaitable
# Mesh is gone (#1367 phase 5). Talon dispatch now uses the A2A
# task-submission path — same wire endpoint as send_a2a_task on
# PeersFeature, but called directly because coordinator dispatch
# happens at the feature level (not from an LLM tool turn).
from kestrel_sovereign.features.talon.runtime import (
    TalonBatchExecution,
    TalonExecution,
    TalonIterateExecution,
    TalonRuntimeError,
    TalonRuntimeRequest,
    build_talon_batch_invocation,
    build_talon_invocation,
    build_talon_iterate_invocation,
    load_talon_policy_preference,
    normalize_auth_lane,
    normalize_backend,
    parse_talon_bool,
    resolve_runtime,
    sanitize_env_for_backend,
    sanitize_untrusted_env,
    write_talon_preference,
)
from kestrel_sovereign.features.talon.github_write import (
    GithubWriteError,
    build_github_write_requests,
    extract_error_message,
    parse_issue_number,
    resolve_write_repo,
)
from kestrel_sovereign.features.talon.verification import (
    CommandExecution,
    TalonVerifier,
    TestCommandResult,
    VerificationEvidence,
    VerificationState,
)
from kestrel_sovereign.waits.engine import MAX_HANDLE_WAIT_SECONDS

logger = logging.getLogger(__name__)


# Default sibling-checkout layout assumed throughout: kestrel-sovereign
# and target repos live as siblings under a common project parent.
_DEFAULT_PROJECT_PARENT = Path(__file__).resolve().parents[4]

# The running agent's own source root. Captured at import time so an
# in-process attacker mutating ``Path(__file__)`` later can't move it.
# The dispatcher refuses to point Talon at this directory; talon
# operates against a separate workspace clone instead. See
# ``_workspace_root_for`` and ``talon_setup_workspace``.
_RUNNING_AGENT_SOURCE_ROOT = Path(__file__).resolve().parents[3]

# Talon's reserved issue-lifecycle labels. kestrel-talon uses
# ``agent-claimed`` as its "this issue is claimed" marker:
# ``GitHubClient.is_claimed()`` returns True iff that label is present,
# and ``kestrel-talon claim`` aborts with "Issue #N is already claimed"
# before doing any work. So a file-then-claim primitive must NOT stamp
# any of these at creation time — Talon applies ``agent-claimed`` itself
# when it claims.
#
# This is the COMPLETE set, pinned as an exact 1:1 mirror of every
# ``label_*`` default in kestrel-talon ``kestreltalon/config.py``:
#
#     label_analyzing  = "agent-analyzing"
#     label_clarifying = "agent-clarifying"
#     label_in_progress = "agent-claimed"   # the is_claimed() marker
#     label_blocked   = "agent-blocked"
#     label_failed    = "agent-failed"
#     label_completed = "agent-complete"
#
# (Note: ``label_in_progress`` resolves to ``agent-claimed`` — there is
# no separate ``agent-in-progress`` label; the claimed marker IS the
# in-progress marker.) Pinned, not imported: ``kestreltalon`` is invoked
# as a CLI binary and is not an importable module in this venv, and
# sibling-checkout path coupling is forbidden (no local-path deps). If
# Talon ever adds/renames a lifecycle label, update this set to match.
_TALON_RESERVED_LABELS = frozenset({
    "agent-analyzing",
    "agent-clarifying",
    "agent-claimed",   # == kestrel-talon label_in_progress
    "agent-blocked",
    "agent-failed",
    "agent-complete",
})


# Orchestrator identity + workflow correlation keys stamped onto every
# outgoing talon invocation at the coordinator boundary (contract with the
# observability team — kestrel-talon#53). Names are FROZEN; do not rename.
#
#   ORCHESTRATOR    — friendly agent name driving this dispatch (unset when
#                     genuinely no agent drove it; downstream renders null
#                     as "Direct").
#   WORKFLOW_RUN_ID — the workflow run id (the driving Signal's session_id
#                     when kind == "workflow.stage"); unset otherwise.
#   STAGE           — the workflow stage name; unset for non-workflow
#                     dispatches.
#
# Transport: CLI dispatches carry them as process env vars on the spawned
# talon process; A2A/AMP dispatches carry the same three keys as structured
# metadata fields on the message. The daemon-side read is kestrel-talon#53.
OBSERVABILITY_ORCHESTRATOR_KEY = "KESTREL_OBSERVABILITY_ORCHESTRATOR"
OBSERVABILITY_WORKFLOW_RUN_ID_KEY = "KESTREL_OBSERVABILITY_WORKFLOW_RUN_ID"
OBSERVABILITY_STAGE_KEY = "KESTREL_OBSERVABILITY_STAGE"


# Per-task suppression of the A2A-preferred claim path (codex P2, detached
# waits). ``dispatch_pipeline(force_cli=True)`` sets this around its
# delegated ``talon_claim`` call so a DETACHED dispatch (the caller keeps a
# ``talon:<job_id>`` wait ref for later) always lands a durable
# ``cli_background`` job: A2A jobs live only in the coordinator's in-memory
# ``_jobs`` map (``_persist_jobs`` persists cli_background rows only), so an
# A2A-shaped wait ref would resolve to an unknown job after an agent
# restart. A ContextVar (not an instance flag) keeps concurrent dispatches
# in the same event loop isolated, and — deliberately — this is NOT a
# ``talon_claim`` tool parameter: the @tool decorator advertises every
# signature param to the LLM, and transport selection is coordinator
# plumbing, not an LLM control.
_FORCE_CLI_DISPATCH: "ContextVar[bool]" = ContextVar(
    "talon_force_cli_dispatch", default=False
)

# Terminal talon job states — a job in one of these has finished (and, on
# success, opened its PR). ``verify_pipeline_ci`` requires a job to be in one
# — settling it first with a fast point-in-time reap, then a bounded wait on
# the durable ``talon:<job_id>`` rail if still running — before binding to the
# PR it produced.
_TERMINAL_TALON_STATES = frozenset(
    {"complete", "failed", "reject", "finished_unknown"}
)

# The one terminal state that means the run actually succeeded (opened its PR
# through a passing pipeline). Every other terminal state is a FAILURE — a PR it
# left behind must not be accepted as CI-green (#2303).
_SUCCESSFUL_TALON_STATE = "complete"

# Non-terminal ("still working") talon job states, the complement of
# ``_TERMINAL_TALON_STATES`` for status bucketing.
_RUNNING_TALON_STATES = frozenset({"dispatched", "running"})

# Observability-backed job status source (#2646). Talon jobs dispatched OUTSIDE
# this coordinator's local registry — driven by a Claude Code session or a peer
# host — never land in ``_jobs``. Those external orchestrators emit job-lifecycle
# telemetry into the shared observability store as ``talon_job`` events. This
# status source reads those events back through the agent's ``observability_store``
# (the per-agent ``a2a_observability`` store, ``ObservabilityStore.query_events``)
# and folds them into per-job status so ``talon_status`` surfaces externally-driven
# jobs that would otherwise be invisible.
#
# Reader-only contract: this coordinator never WRITES ``talon_job`` events — the
# external orchestrator (Claude Code session, peer host, kestrel-talon) is the
# producer. A stock single-agent deployment whose store holds no ``talon_job``
# rows simply reports registry jobs only, which is expected, not a bug.
_TALON_JOB_EVENT_TYPE = "talon_job"

# Provenance markers stamped on every ``talon_status`` job entry (#2646) so a
# consumer can tell a locally-dispatched job from one only seen via observability.
JOB_SOURCE_REGISTRY = "registry"
JOB_SOURCE_OBSERVABILITY = "observability"

# Accepted ``source`` filter values for ``talon_status`` (#2646). ``None``/""
# and ``"all"`` merge both provenances; the registry/observability aliases
# restrict to one. An unrecognized value is REJECTED (rather than silently
# returning an empty set) so a typo'd filter is discoverable — mirroring the
# scheduler's task-name validation.
_SOURCE_FILTER_ALL = frozenset({"", "all"})
_SOURCE_FILTER_REGISTRY = frozenset({"registry", "local"})
_SOURCE_FILTER_OBSERVABILITY = frozenset(
    {"observability", "external", "observed"}
)

# Default lookback window / row cap when reducing observability events into
# per-job status. A talon run takes 10-30 min, so a day of lookback comfortably
# covers in-flight and recently-finished external jobs without an unbounded scan.
_OBSERVABILITY_LOOKBACK_MINUTES = 1440
_OBSERVABILITY_EVENT_LIMIT = 1000


def _talon_event_status(phase: Any) -> str:
    """Map a ``talon_job`` event's lifecycle ``talon_event`` phase to a status.

    kestrel-talon emits ``claimed`` / ``started`` / ``iteration`` while a run is in
    flight and ``completed`` / ``failed`` / ``rejected`` as terminal markers. The
    store returns events newest-first (``ORDER BY timestamp DESC``), so the latest
    event per job defines the current status — this never invents a terminal state
    from a still-running stream.
    """
    p = phase.strip().lower() if isinstance(phase, str) else ""
    if p == "completed":
        return "complete"
    if p == "failed":
        return "failed"
    if p in ("rejected", "reject"):
        return "reject"
    # claimed / started / iteration / anything non-terminal → still in flight.
    return "running"


def _coerce_issue(raw: Any) -> Optional[int]:
    """Best-effort int coercion of a ``talon_job`` event's ``issue`` field."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return int(raw.strip())
    return None


_PR_NUMBER_FROM_URL_RE = re.compile(r"/pull/(\d+)")

# Bounded commands retain only a modest tail from each stream. Every public
# Talon surface exposes at most 2,000 characters of evidence, so a 64 KiB byte
# cap leaves ample UTF-8/traceback context without letting a hostile test or git
# filter grow the coordinator process without bound.
_TALON_CAPTURE_LIMIT_BYTES = 65_536


def _pr_number_from_url(url: Optional[str]) -> Optional[int]:
    """Parse the PR number out of a ``.../pull/<n>`` GitHub URL, or None."""
    if not url:
        return None
    match = _PR_NUMBER_FROM_URL_RE.search(url)
    return int(match.group(1)) if match else None


# Discriminated reason codes for ``talon_file_and_claim`` failures.
# Each one names a distinct fix path the agent (or the operator) can act
# on. They're public values, not just internal hints — every failed
# ToolResult carries one in ``data['reason_code']`` AND in the top-level
# error string, so the model's narration can surface "MISSING_GH_AUTH —
# please authenticate" instead of the historical catch-all "may have
# been denied at the approval gate, gh is not authenticated, or it
# failed for a non-label reason" (#1383).
TALON_FAC_REASONS = {
    "MISSING_COMPUTER_USE",  # ComputerUseFeature not enabled
    "GATE_DENIED",           # approval gate refused the shell
    "MISSING_GH_AUTH",       # `gh` not authenticated / no token
    "GH_NOT_INSTALLED",      # the `gh` binary itself is absent
    "REPO_NOT_FOUND",        # repo not visible to the auth'd user
    "LABEL_REJECTED",        # label unknown/invalid even after retry
    "SHELL_TIMEOUT",         # shell call hit the 120s wall
    "URL_PARSE_FAILED",      # gh exited 0 but URL regex missed
    "DISPATCH_FAILED",       # talon_claim never reported dispatched
    "UNKNOWN_FAILURE",       # fallthrough
}


def _gh_failure_reason(
    shell_res,
    stdout: str,
    stderr: str,
    *,
    succeeded: bool,
    parsed_url: bool,
) -> tuple[str, str]:
    """Classify a ``gh issue create`` failure into a stable reason code.

    The shell call goes through ``ComputerUseFeature.shell``, which
    returns:
      * ``ok``      — rc=0  (stdout/stderr on ``data``)
      * ``partial`` — ran but rc!=0 (caveat in ``error``; ``data`` has rc/timeout)
      * ``failed``  — gate denied / backend exception / empty argv

    We sniff stderr/stdout/error for the patterns the actual ``gh`` CLI
    emits (and the gate's ``denied_reason``) and pick the most specific
    code. The returned ``(code, hint)`` tuple is rendered into the
    ``ToolResult.failed`` error string so the agent's narration carries
    something actionable.
    """
    error_blob = (shell_res.error or "") if shell_res else ""
    full = f"{stderr}\n{stdout}\n{error_blob}".lower()
    timed_out = bool(
        getattr(shell_res, "data", None)
        and shell_res.data.get("timed_out")
    )

    if timed_out:
        return (
            "SHELL_TIMEOUT",
            "`gh issue create` exceeded the 120s wall. Retry, or check "
            "network reachability to api.github.com.",
        )

    # Stderr/stdout patterns are checked BEFORE shell-status branches so
    # that a non-zero exit (partial) with "Bad credentials" in stderr is
    # classified as MISSING_GH_AUTH instead of falling through to a
    # generic UNKNOWN_FAILURE. The patterns are ordered most-specific
    # first.
    if "gh: command not found" in full or (
        "no such file" in full and "gh" in full
    ):
        return (
            "GH_NOT_INSTALLED",
            "the `gh` CLI is not on PATH for this agent. Install gh in "
            "the host environment.",
        )

    if (
        "bad credentials" in full
        or "gh auth login" in full
        or "no github token" in full
        or "authentication required" in full
        or "http 401" in full
        or " 401" in full
        or "401 unauthorized" in full
    ):
        return (
            "MISSING_GH_AUTH",
            "`gh` is not authenticated. Set GH_TOKEN / GITHUB_TOKEN, or "
            "run `gh auth login` in the host environment.",
        )

    if (
        "could not resolve to a repository" in full
        or ("not found" in full and "repository" in full)
        or "http 404" in full
        or "404 not found" in full
    ):
        return (
            "REPO_NOT_FOUND",
            "GitHub returned 404 / not-found for the repo. Verify the "
            "owner/name and that the auth'd user has access.",
        )

    if "label" in full and (
        "not found" in full
        or "could not add" in full
        or "not a valid" in full
        or "no label" in full
    ):
        return (
            "LABEL_REJECTED",
            "gh rejected one of the labels even after the no-label retry. "
            "Check that the repo accepts the labels you passed.",
        )

    if shell_res is not None and str(shell_res.status) in (
        "error", "failed", "ToolResultStatus.ERROR",
    ):
        # Backend exception, empty argv, or gate denial — surface via
        # ``shell_res.error`` when ``data`` was empty. (ToolResult.failed
        # maps to status "error"; we accept "failed" too so test fakes
        # that pass the historical string still classify correctly.)
        elow = error_blob.lower()
        if "denied" in elow or "approval" in elow or "policy" in elow:
            return (
                "GATE_DENIED",
                "the approval-gate denied the shell. Allowlist the "
                "`gh issue create` pattern or escalate to the operator.",
            )
        if "empty command" in elow:
            return (
                "GH_NOT_INSTALLED",
                "shell argv was empty — likely shlex parse failure.",
            )

    if succeeded and not parsed_url:
        return (
            "URL_PARSE_FAILED",
            "`gh issue create` exited 0 but stdout did not contain a "
            "parseable issue URL. The repo may use an unusual gh config.",
        )

    return (
        "UNKNOWN_FAILURE",
        "shell ran but did not produce a parseable issue URL. Inspect "
        "stderr_tail / stdout_tail / shell_error in the structured data.",
    )


def _path_contains(parent: Path, child: Path) -> bool:
    """True iff ``child`` is ``parent`` or under it. Resolves both."""
    try:
        parent_r = parent.resolve()
        child_r = child.resolve()
    except OSError:
        return False
    if parent_r == child_r:
        return True
    return parent_r in child_r.parents


class _VerifyCwdError(Exception):
    """Raised when a verify cwd can't be resolved to a safe workspace.

    Carries the structured ``data`` payload ``talon_verify`` returns so the
    refusal (e.g. ``workspace_not_provisioned``) reaches the agent intact.
    """

    def __init__(self, message: str, data: Dict[str, Any]):
        super().__init__(message)
        self.data = data


class TalonCoordinatorFeature(Feature):
    """Thin dispatcher to the external kestrel-talon daemon.

    Provides !talon commands for claiming issues, batch processing,
    and checking status. Prefers mesh dispatch over CLI fallback.
    """

    def __init__(self, agent):
        super().__init__(agent)
        # message_id -> {pid, started_at, log_path, command, repo,
        # issue, status, returncode, completed_at, process}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # workflow run_id -> the talon job_id THAT run's talon_run stage
        # dispatched. Lets ``verify_pipeline_ci`` bind CI verification to this
        # run's own job instead of a repo/issue correlation that a concurrent
        # run (e.g. a retry for the same issue) could poison (#2303).
        self._pipeline_run_jobs: Dict[str, str] = {}
        # Eager reload so a fresh feature instance immediately sees
        # jobs from a previous process — dispatch-then-persist would
        # otherwise truncate the registry to the new job alone.
        self._reload_persisted_jobs()

    @property
    def tool_description(self) -> str:
        return "Dispatch work to Talon autonomous coding agent and monitor status"

    async def initialize(self):
        logger.info("TalonCoordinatorFeature initialized")
        # Self-register the talon.job_complete signal source on the
        # agent's signal registry. Owning the registration here (vs.
        # in agent boot) keeps it scoped to this feature so when
        # Talon eventually extracts to an external feature package
        # the registration travels with it. No-op if signal_registry
        # is absent (test stubs, headless agents) — and idempotent
        # on a second initialize() since a duplicate registration
        # would otherwise raise and shadow real-failure warnings.
        registry = getattr(self.agent, "signal_registry", None)
        if registry is not None and hasattr(registry, "register_with_policy"):
            from kestrel_sovereign.signals import RegistrationPolicy
            from kestrel_sovereign.signals.sources.talon import (
                build_talon_job_complete_registration,
            )
            # OPTIONAL policy (#2522): idempotent on a second initialize(); a
            # talon.job_complete already present with a DIFFERENT contract is
            # reported rather than silently accepted by a precheck-by-name skip.
            # Never raises — one bad source must not abort feature init.
            # Own the sources we newly register (here and below) so the
            # base-class shutdown / boot rollback unregisters exactly them and
            # never a host's pre-existing source (#2522 P2).
            self._own_signal_sources(
                registry.register_with_policy(
                    build_talon_job_complete_registration(),
                    RegistrationPolicy.OPTIONAL,
                )
            )

            # Register the host-provided sources the built-in
            # ``stalled_work_rescue`` workflow references (#2192). The rescue
            # loop is Talon/fleet-coordination domain — detect stalled work,
            # dispatch repairs to Talon, verify evidence, close todos — so the
            # coordinator owns these agent-native registrations. Without them
            # the workflow runner's start-contract validation fails before a run
            # record is created ("references unregistered source"). Idempotent:
            # a source a host already registered with a richer implementation is
            # left untouched.
            from kestrel_sovereign.signals.sources.workflow_rescue import (
                register_workflow_rescue_sources,
            )
            # Bind live discovery so a recurring fleet_stalled_sweep tick
            # surveys real stalled Talon jobs instead of relying on pre-seeded
            # candidates (#2200). Read-only: it only observes, never dispatches.
            registered = register_workflow_rescue_sources(
                registry,
                fleet_stalled_discover=self._survey_stalled_talon_jobs,
            )
            self._own_signal_sources(registered)
            if registered:
                logger.info(
                    "TalonCoordinatorFeature registered workflow-rescue "
                    "signal sources: %s", ", ".join(registered),
                )

            # Register the purpose-built talon pipeline dispatch source so
            # workflow stages can ACTUALLY dispatch a full kestrel-talon
            # pipeline run (claim/iterate) and resolve on completion via the
            # talon.job_complete / TalonWaitable rail. Bound to this feature
            # because the dispatch plumbing (A2A preferred, CLI fallback,
            # observability stamping) lives here. Idempotent like the rest.
            from kestrel_sovereign.signals.sources.talon_pipeline import (
                register_talon_pipeline_source,
            )
            if register_talon_pipeline_source(registry, self):
                from kestrel_sovereign.signals.sources.talon_pipeline import (
                    SOURCE_NAME as _TALON_PIPELINE_SOURCE,
                )

                self._own_signal_sources(_TALON_PIPELINE_SOURCE)
                logger.info(
                    "TalonCoordinatorFeature registered the "
                    "talon_pipeline_dispatch signal source",
                )

            # Register the support sources the built-in ``fleet_coding_pipeline``
            # workflow names (#2303) and inject the workflow itself into the
            # workflows package's built-in registry so workflow_load_builtin /
            # workflow_run can load and start it. Fleet-coordination domain, so
            # the coordinator owns these registrations (same posture as the
            # workflow-rescue sources above). Both calls are idempotent and the
            # built-in injection no-ops when kestrel-feature-workflows is absent.
            from kestrel_sovereign.signals.sources.fleet_coding_pipeline import (
                register_fleet_coding_pipeline_builtin,
                register_fleet_coding_pipeline_sources,
            )
            fleet_registered = register_fleet_coding_pipeline_sources(
                registry, coordinator=self
            )
            self._own_signal_sources(fleet_registered)
            if fleet_registered:
                logger.info(
                    "TalonCoordinatorFeature registered fleet-coding-pipeline "
                    "signal sources: %s", ", ".join(fleet_registered),
                )
            if register_fleet_coding_pipeline_builtin():
                logger.info(
                    "TalonCoordinatorFeature registered the fleet_coding_pipeline "
                    "built-in workflow",
                )

    async def post_all_features_loaded(self, agent):
        """Register the ``talon:`` Waitable provider with the wait engine.

        Lets the generic ``wait("talon:<job_id>")`` tool dispatch here, and
        lets the wait reconciler enumerate in-flight jobs for auto-wake /
        signal-resume.
        """
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            registry.register(TalonWaitable(self), replace=True)

        # One-time migration of legacy talon_monitor dedup state into the
        # generic reconciler ledger. Pre-Wave-2 jobs.json rows carry
        # ``last_signaled_status`` from the retired monitor; without seeding
        # the (initially empty) ledger, the first wait_reconcile tick would
        # re-fire talon.job_complete for every already-delivered terminal job
        # (codex Wave 2 P2). seed_signaled is INSERT-OR-IGNORE, so this is a
        # safe no-op on every subsequent startup.
        await self._seed_legacy_signal_ledger(agent)

    async def _seed_legacy_signal_ledger(self, agent) -> None:
        """Seed wait_signal_state from legacy jobs.json ``last_signaled_status``."""
        raw_storage = getattr(agent, "_raw_storage", None)
        db = getattr(raw_storage, "db", None)
        if db is None:
            return
        try:
            from kestrel_sdk.tools import Outcome
            from kestrel_sovereign.storage.async_wait_signal_store import (
                WaitSignalStore,
            )

            agent_id = (
                getattr(agent, "did", None)
                or getattr(agent, "agent_id", None)
                or ""
            )
            store = WaitSignalStore(db, str(agent_id))
            self._reload_persisted_jobs()
            for job_id, info in self._jobs.items():
                legacy = info.get("last_signaled_status")
                if not legacy:
                    continue
                # The reconciler dedups on a token of the generic Outcome
                # PLUS the provider's native status ("<outcome>:<status>"), so
                # finished_unknown vs failed (both FAILED) stay distinct. Seed
                # the SAME token shape from the legacy talon status, else the
                # first tick would see a mismatch and re-fire (complete -> done;
                # the rest -> failed).
                outcome = (
                    Outcome.DONE if legacy == "complete" else Outcome.FAILED
                )
                token = f"{outcome.value}:{legacy}"
                await store.seed_signaled("talon", str(job_id), token)
        except Exception as e:  # never let a migration hiccup block startup
            logger.warning(
                "TalonCoordinatorFeature: legacy signal-ledger seed failed: %s",
                e,
            )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        name="scan_stale_work",
        description=(
            "Read-only ecosystem discovery scan for stale Talon work. "
            "Returns actionable findings for scheduler discovery watches; "
            "never dispatches repairs or closes issues."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon scan-stale-work",
    )
    async def scan_stale_work(
        self,
        stale_days: int = 3,
        repo: Optional[str] = None,
        repos: Optional[Any] = None,
        org: Optional[Any] = None,
        repo_prefix: Optional[Any] = None,
        exclude_repos: Optional[Any] = None,
    ) -> ToolResult:
        """Scan for stale work without taking action (#2281, #2269).

        This is the scheduler-facing wrapper around Talon's existing live
        stalled-job survey. It keeps ``ecosystem_discovery_watch`` wired to a
        real in-tree tool while preserving the evidence boundary: discovery
        returns findings only, and any repair/closure still needs a later gate.

        Two modes:

        * **Single-repo / no filter (legacy).** With no roster args, it surveys
          all live stalled Talon jobs, optionally filtered to one ``repo`` slug.
        * **Roster mode (#2269).** When any of ``repos`` / ``org`` /
          ``repo_prefix`` is given — or ``repo`` carries a wildcard — the args
          are parsed into a durable ecosystem roster and expanded against the
          repos accessible to the agent's GitHub token. Wildcards like
          ``KestrelSovereignAI/kestrel-feature-*`` are treated as prefixes, not
          literal repo names; tekspear repos are always excluded; and repos that
          could not be resolved are reported as explicit ``scan_failures``.

        Args:
            stale_days: How many idle days mark Talon work as stalled.
            repo: Optional ``owner/name`` filter (or a wildcard → roster mode).
            repos: Explicit allowlist of ``owner/name`` slugs (or wildcards).
            org: Org(s) whose accessible repos form the roster.
            repo_prefix: Prefix(es) matched against accessible repos.
            exclude_repos: Explicit ``owner/name`` slugs to drop from the roster.
        """
        from kestrel_sovereign.signals.sources.ecosystem_roster import (
            is_wildcard,
            parse_roster_spec,
            expand_roster,
        )

        roster_requested = bool(repos or org or repo_prefix) or bool(
            repo and is_wildcard(str(repo))
        )

        stalled = await self._survey_stalled_talon_jobs(stale_days)

        if not roster_requested:
            if repo:
                stalled = [item for item in stalled if item.get("repo") == repo]
            findings = self._stale_findings(stalled, default_repo=repo or "")
            return ToolResult.ok(
                confirmation=(
                    f"Found {len(findings)} stale work item(s) "
                    f"older than {stale_days} day(s)."
                ),
                data={
                    "summary": (
                        f"{len(findings)} stale work item(s)"
                        if findings
                        else "No actionable stale work findings."
                    ),
                    "findings": findings,
                    "stale_days": stale_days,
                    "repo": repo or "",
                },
            )

        spec = parse_roster_spec(
            org=org,
            repos=repos,
            repo=repo,
            repo_prefix=repo_prefix,
            exclude_repos=exclude_repos,
        )
        accessible, discovery_error = await self._discover_roster_universe(spec.orgs)
        expansion = expand_roster(
            spec,
            accessible_repos=accessible,
            discovery_error=discovery_error,
        )

        roster = set(expansion.repos)
        findings = self._stale_findings(
            [item for item in stalled if item.get("repo") in roster],
        )

        scan_failures: List[Dict[str, Any]] = []
        for failure in expansion.failures:
            target = (
                failure.get("repo")
                or failure.get("pattern")
                or failure.get("scope")
                or "?"
            )
            entry = {
                "repo": failure.get("repo", ""),
                "kind": "scan_failure",
                "status": "inaccessible",
                "severity": "high",
                "title": f"Roster scan failure: {target} ({failure.get('reason', '')})",
                "reason": failure.get("reason", ""),
                "target": target,
                "suggested_gate": "triage_lane",
                "actionable": True,
            }
            scan_failures.append(entry)
            # Surface failures as findings too so the discovery watch never
            # silently drops an inaccessible repo (#2269 AC3).
            findings.append(entry)

        summary_bits = []
        if findings:
            summary_bits.append(f"{len(findings)} finding(s)")
        summary_bits.append(f"{len(expansion.repos)} repo(s) scanned")
        if scan_failures:
            summary_bits.append(f"{len(scan_failures)} scan failure(s)")

        return ToolResult.ok(
            confirmation=(
                f"Scanned {len(expansion.repos)} roster repo(s) for stale work "
                f"older than {stale_days} day(s); "
                f"{len(scan_failures)} inaccessible."
            ),
            data={
                "summary": (
                    ", ".join(summary_bits)
                    if findings or scan_failures
                    else "No actionable stale work findings."
                ),
                "findings": findings,
                "scan_failures": scan_failures,
                "scanned_repos": list(expansion.repos),
                "excluded_repos": list(expansion.excluded),
                "stale_days": stale_days,
                "repo": repo or "",
            },
        )

    def _stale_findings(
        self,
        stalled: List[Dict[str, Any]],
        default_repo: str = "",
    ) -> List[Dict[str, Any]]:
        """Shape surveyed stalled Talon jobs into actionable discovery findings."""
        findings: List[Dict[str, Any]] = []
        for item in stalled:
            issue = item.get("issue")
            repo_name = item.get("repo") or default_repo or ""
            title_parts = ["Stalled Talon job", str(item.get("id") or "?")]
            if repo_name and issue:
                title_parts.append(f"for {repo_name}#{issue}")
            findings.append({
                "id": item.get("id"),
                "repo": repo_name,
                "kind": item.get("kind") or "talon_job",
                "issue": issue,
                "job": item.get("id"),
                "status": item.get("status") or "stalled",
                "severity": "high",
                "title": " ".join(title_parts),
                "started_at": item.get("started_at"),
                "suggested_gate": "govern_stalled_work_rescue",
                "actionable": True,
            })
        return findings

    async def _discover_roster_universe(
        self, orgs: tuple[str, ...]
    ) -> tuple[set[str], Optional[str]]:
        """Fetch the set of repos accessible to the agent's GitHub token.

        Returns ``(accessible_repos, error)``. ``error`` is set when a listing
        could not be fetched at all; a partial set with an error means some orgs
        listed and others did not — :func:`expand_roster` turns the missing ones
        into explicit failures. ``orgs`` empty falls back to the configured
        default orgs (``[github].orgs`` or ``KestrelSovereignAI``).
        """
        from kestrel_sovereign.endpoints.github import discover_accessible_repos

        accessible: set[str] = set()
        error: Optional[str] = None
        targets: List[Optional[str]] = list(orgs) if orgs else [None]
        for target in targets:
            try:
                found = await discover_accessible_repos(org=target)
                accessible.update(found)
            except Exception as exc:  # noqa: BLE001 - inaccessible → explicit failure
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "scan_stale_work: could not list repos for org %s: %s",
                    target or "<default>",
                    exc,
                )
        return accessible, error

    @tool(
        name="talon_claim",
        description=(
            "Dispatch a single issue to Talon for autonomous "
            "implementation. Returns immediately with a job_id; the "
            "actual work runs in the background. Poll talon_status "
            "or talon_job_log to follow progress."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon claim",
    )
    async def talon_claim(
        self,
        repo: str,
        issue: int,
        max_iterations: Optional[int] = None,
        max_turns: Optional[int] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        auth_lane: Optional[str] = None,
        skip_clarification: Optional[bool] = None,
        worktree: bool = True,
        self_review: Optional[bool] = None,
        demo_check: Optional[bool] = None,
        eye_check: Optional[bool] = None,
    ) -> ToolResult:
        """Claim an issue for Talon to implement.

        Talon's claim flow runs the LLM agent loop, runs quality
        gates, commits, pushes, and opens a PR — typically 10-30
        minutes per issue. This dispatcher launches it in the
        background and returns immediately. The previous synchronous
        implementation waited up to 300s for completion and killed
        Talon mid-implementation; that's why earlier dispatches
        looked like they "did nothing."

        Args:
            repo: GitHub repo in owner/name format
                (e.g. ``KestrelSovereignAI/kestrel-sovereign``). The
                special string ``"self"`` resolves to
                ``KestrelSovereignAI/kestrel-sovereign``.
            issue: Issue number to claim.
            max_iterations: Max LLM implementation iterations. If unset,
                uses ``[talon.preference].max_iterations``.
            max_turns: Max agent turns per Talon iteration. If unset,
                uses ``[talon.preference].max_turns``.
            backend: Talon runtime backend: ``claude``, ``codex``, or
                ``opencode``. This is separate from Kestrel chat LLM routing.
            model: Backend-specific model. Claude accepts ``opus``,
                ``sonnet``, or ``haiku``; Codex accepts current Codex model
                IDs; OpenCode accepts provider/model IDs.
            auth_lane: ``oauth``, ``api_key``, or ``provider_config``.
            skip_clarification: If True, skip the analysis/clarification
                phase. Recommended when nobody is watching to answer
                questions; the issue body should already be specific.
            worktree: If True, run in an isolated git worktree so the
                target checkout isn't clobbered. README marks this
                "strongly recommended" — don't disable unless you
                know what you're doing.
            demo_check: If True, run kestrel-talon's demo gate
                (``--demo-check``). CLI-only — requesting it forces the
                CLI dispatch path since A2A dispatch carries repo/issue
                only.
            eye_check: If True, run kestrel-eye visual verification
                (``--eye-check``). CLI-only, same as demo_check.

        Returns:
            ``{"dispatched": True, "method": "cli_background",
            "job_id": ..., "log_path": ..., "pid": ...}`` on success.
            Failure returns ``{"dispatched": False, "error": ...}``.
        """
        try:
            policy, preference = load_talon_policy_preference()
            runtime_request = TalonRuntimeRequest(
                backend=normalize_backend(backend),
                model=model,
                auth_lane=normalize_auth_lane(auth_lane),
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "invalid_talon_runtime",
                    "error": str(e),
                },
            )

        try:
            resolved_backend, resolved_model, resolved_auth_lane = resolve_runtime(
                runtime_request,
                preference,
                policy,
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "talon_policy_rejected",
                    "error": str(e),
                },
            )

        # A2A dispatch only carries repo/issue today, so ANY explicitly
        # provided per-run control — runtime (backend/model/auth_lane),
        # iteration caps, clarification/self-review behavior, quality
        # gates, or a worktree opt-out — would be silently dropped on that
        # path and the daemon's defaults would apply instead (codex P2).
        # Every explicit flag therefore forces the CLI invocation, which
        # carries them all. Do NOT widen the A2A payload here — that is
        # the daemon team's side of the contract.
        use_a2a = (
            resolved_backend == "claude"
            and resolved_model == "opus"
            and resolved_auth_lane == "oauth"
            and backend is None
            and model is None
            and auth_lane is None
            and max_iterations is None
            and max_turns is None
            and skip_clarification is None
            and self_review is None
            and worktree is True
            and demo_check is None
            and eye_check is None
            # Detached dispatches (dispatch_pipeline force_cli) need a
            # durable cli_background job — A2A jobs are in-memory only
            # and their wait refs die with the process.
            and not _FORCE_CLI_DISPATCH.get()
        )
        if use_a2a:
            a2a_result = await self._dispatch_via_a2a(repo, issue)
            if a2a_result.get("dispatched"):
                tracking_id = (
                    a2a_result.get("job_id")
                    or a2a_result.get("task_id")
                    or "?"
                )
                return ToolResult.ok(
                    confirmation=(
                        f"Dispatched {repo}#{issue} to talon via A2A "
                        f"(task_id={tracking_id})"
                    ),
                    data=a2a_result,
                )

        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = (
            self._assert_workspace_safe(workspace)
            if policy.require_sandboxed_workspace
            else None
        )
        if unsafe_reason:
            return ToolResult.failed(
                unsafe_reason,
                data={
                    "dispatched": False,
                    "state": "unsafe_workspace",
                    "error": unsafe_reason,
                },
            )

        # If the workspace doesn't exist, refuse and tell the agent
        # the structural next step. Don't silently fall through to
        # the running source tree — that's the bug we're fixing.
        state = self._workspace_state(repo_resolved)
        if not state["exists"] or not state["is_git"]:
            err_msg = (
                "No talon workspace exists for "
                f"{repo_resolved} at {workspace}. The dispatcher "
                "will not operate on the running agent's source "
                "tree. Call talon_setup_workspace(repo) to "
                "provision a sandboxed clone, then retry."
            )
            return ToolResult.failed(
                err_msg,
                data={
                    "dispatched": False,
                    "state": "workspace_not_provisioned",
                    "error": err_msg,
                    "workspace": state,
                    "next_step": (
                        f"talon_setup_workspace(repo='{repo_resolved}')"
                    ),
                },
            )

        worktree_base = (
            os.environ.get("KESTREL_TALON_WORKTREE_BASE")
            or str(workspace.parent)
        )

        try:
            execution = TalonExecution(
                repo=repo_resolved,
                issue=issue,
                repo_dir=workspace,
                worktree_base=Path(worktree_base),
                worktree=worktree,
                max_iterations=(
                    int(max_iterations)
                    if max_iterations is not None
                    else preference.max_iterations
                ),
                max_turns=(
                    int(max_turns)
                    if max_turns is not None
                    else preference.max_turns
                ),
                skip_clarification=(
                    parse_talon_bool(skip_clarification, "skip_clarification")
                    if skip_clarification is not None
                    else preference.skip_clarification
                ),
                self_review=(
                    parse_talon_bool(self_review, "self_review")
                    if self_review is not None
                    else preference.self_review
                ),
                demo_check=(
                    parse_talon_bool(demo_check, "demo_check")
                    if demo_check is not None
                    else False
                ),
                eye_check=(
                    parse_talon_bool(eye_check, "eye_check")
                    if eye_check is not None
                    else False
                ),
            )
            invocation = build_talon_invocation(
                runtime_request,
                execution,
                policy=policy,
                preference=preference,
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "talon_policy_rejected",
                    "error": str(e),
                },
            )

        cli_result = await self._dispatch_via_cli_background(
            invocation.argv,
            label=f"claim:{repo_resolved}#{issue}",
            env=invocation.env,
            extra_meta={
                "repo": repo_resolved,
                "issue": issue,
                "workspace": str(workspace),
                **invocation.metadata(),
            },
        )

        if cli_result.get("dispatched"):
            return ToolResult.ok(
                confirmation=(
                    f"Dispatched {repo_resolved}#{issue} to talon via CLI "
                    f"background (job_id={cli_result.get('job_id', '?')}, "
                    f"pid={cli_result.get('pid', '?')})"
                ),
                data=cli_result,
            )
        return ToolResult.failed(
            cli_result.get("error") or "talon CLI dispatch failed",
            data=cli_result,
        )

    async def dispatch_pipeline(
        self,
        *,
        repo: str,
        issue: Optional[int] = None,
        pr: Optional[int] = None,
        mode: str = "claim",
        self_review: Optional[bool] = None,
        demo_check: bool = False,
        eye_check: bool = False,
        force_cli: bool = False,
    ) -> Dict[str, Any]:
        """Dispatch a full talon pipeline run (claim or iterate).

        The dispatch seam behind the ``talon_pipeline_dispatch`` signal
        source: NOT an LLM tool. Claim mode delegates to :meth:`talon_claim`
        so it inherits the A2A-preferred/CLI-fallback plumbing, policy
        enforcement, and workspace safeguards unchanged. Iterate mode
        builds a ``kestrel-talon iterate`` invocation through the same
        runtime policy layer and the same background-CLI funnel — which
        also stamps the observability keys (kestrel-talon#53).

        ``force_cli=True`` suppresses the A2A-preferred claim path for this
        call (via the task-local ``_FORCE_CLI_DISPATCH`` override). Callers
        that hand back a ``talon:<job_id>`` wait ref for later use (the
        source's detached ``wait: false`` mode) MUST set it: A2A jobs live
        only in the in-memory ``_jobs`` map (``_persist_jobs`` persists
        cli_background rows only), so an A2A wait ref would resolve to an
        unknown job after an agent restart.

        Returns a plain dispatch dict (``{"dispatched": bool, ...}`` with
        ``job_id``/``task_id`` on success), never a ToolResult, so signal
        handlers can fail closed on it directly.
        """
        if mode == "claim":
            if issue is None:
                return {
                    "dispatched": False,
                    "error": "dispatch_pipeline: claim mode requires issue",
                }
            token = _FORCE_CLI_DISPATCH.set(True) if force_cli else None
            try:
                claim = await self.talon_claim(
                    repo=repo,
                    issue=int(issue),
                    self_review=self_review,
                    demo_check=demo_check or None,
                    eye_check=eye_check or None,
                )
            finally:
                if token is not None:
                    _FORCE_CLI_DISPATCH.reset(token)
            data = dict(claim.data or {})
            if not data.get("dispatched") and claim.error:
                data.setdefault("error", claim.error)
            self._record_pipeline_run_job(data)
            return data

        if mode != "iterate":
            return {
                "dispatched": False,
                "error": f"dispatch_pipeline: unknown mode {mode!r}",
            }
        if pr is None:
            return {
                "dispatched": False,
                "error": "dispatch_pipeline: iterate mode requires pr",
            }

        try:
            policy, preference = load_talon_policy_preference()
        except TalonRuntimeError as e:
            return {
                "dispatched": False,
                "state": "invalid_talon_runtime",
                "error": str(e),
            }

        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = (
            self._assert_workspace_safe(workspace)
            if policy.require_sandboxed_workspace
            else None
        )
        if unsafe_reason:
            return {
                "dispatched": False,
                "state": "unsafe_workspace",
                "error": unsafe_reason,
            }

        state = self._workspace_state(repo_resolved)
        if not state["exists"] or not state["is_git"]:
            err_msg = (
                f"No talon workspace exists for {repo_resolved} at "
                f"{workspace}. The dispatcher will not operate on the "
                "running agent's source tree. Call "
                "talon_setup_workspace(repo) to provision a sandboxed "
                "clone, then retry."
            )
            return {
                "dispatched": False,
                "state": "workspace_not_provisioned",
                "error": err_msg,
                "workspace": state,
                "next_step": f"talon_setup_workspace(repo='{repo_resolved}')",
            }

        worktree_base = (
            os.environ.get("KESTREL_TALON_WORKTREE_BASE")
            or str(workspace.parent)
        )

        try:
            execution = TalonIterateExecution(
                repo=repo_resolved,
                pr=int(pr),
                repo_dir=workspace,
                worktree_base=Path(worktree_base),
                worktree=True,
                max_turns=preference.max_turns,
                self_review=(
                    parse_talon_bool(self_review, "self_review")
                    if self_review is not None
                    else preference.self_review
                ),
                demo_check=bool(demo_check),
                eye_check=bool(eye_check),
            )
            invocation = build_talon_iterate_invocation(
                TalonRuntimeRequest(),
                execution,
                policy=policy,
                preference=preference,
            )
        except TalonRuntimeError as e:
            return {
                "dispatched": False,
                "state": "talon_policy_rejected",
                "error": str(e),
            }

        dispatch = await self._dispatch_via_cli_background(
            invocation.argv,
            label=f"iterate:{repo_resolved}#{pr}",
            env=invocation.env,
            extra_meta={
                "repo": repo_resolved,
                "pr": int(pr),
                "workspace": str(workspace),
                **invocation.metadata(),
            },
        )
        self._record_pipeline_run_job(dispatch)
        return dispatch

    def _current_workflow_run_id(self) -> Optional[str]:
        """The in-flight workflow run id (the driving Signal's ``session_id``).

        Both the ``talon_run`` and ``verify_ci`` stages of a
        ``fleet_coding_pipeline`` run dispatch through this coordinator inside
        that run's signal context, so this is a stable key for pairing the CI
        verification with the job the run's own dispatch produced (#2303).
        Returns None outside a signal dispatch.
        """
        try:
            from kestrel_sovereign.signals.context import get_current_signal

            signal = get_current_signal()
        except Exception:  # pragma: no cover - defensive import guard
            return None
        run_id = getattr(signal, "session_id", None) if signal is not None else None
        return run_id if isinstance(run_id, str) and run_id else None

    def _record_pipeline_run_job(self, dispatch: Any) -> None:
        """Bind the current workflow run to the talon job it just dispatched.

        Durable with the same guarantees as the job registry (#2303, fifth
        pass): besides the in-memory ``_pipeline_run_jobs`` map, the binding is
        stamped onto the persisted job record (``workflow_run_id``) and flushed
        via :meth:`_persist_jobs`. :meth:`_reload_persisted_jobs` reconstructs
        the map from those records on boot, so a restart between the
        ``talon_run`` and ``verify_ci`` stages still resolves this run's own job
        instead of failing closed on a lost in-memory map.
        """
        if not isinstance(dispatch, dict) or not dispatch.get("dispatched"):
            return
        job_id = dispatch.get("job_id") or dispatch.get("task_id")
        run_id = self._current_workflow_run_id()
        if not (isinstance(job_id, str) and job_id and run_id):
            return
        self._pipeline_run_jobs[run_id] = job_id
        info = self._jobs.get(job_id)
        if isinstance(info, dict):
            # Persist the binding on the job record so it survives a restart
            # (only cli_background rows are persisted — exactly the durable
            # ``wait: false`` dispatch path this pipeline uses).
            info["workflow_run_id"] = run_id
            self._persist_jobs()

    async def verify_pipeline_ci(
        self,
        *,
        repo: Optional[str] = None,
        run_id: Optional[str] = None,
        poll_interval_seconds: int = 15,
        max_ci_wait_seconds: int = 1800,
        max_job_wait_seconds: int = MAX_HANDLE_WAIT_SECONDS,
        required_checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Confirm the CI of the PR THIS pipeline's talon run produced is green.

        The verify seam behind the ``fleet_ci_probe`` signal source (#2303).
        Verification is bound to the *dispatched run's own output* — never a
        caller-supplied branch, job id, or repo/issue correlation. It resolves
        the talon job **exclusively** from the coordinator's run_id→job_id map
        (recorded at dispatch time by :meth:`_record_pipeline_run_job`), keyed
        by the workflow run's ``session_id``. The workflows runner sets that
        ``session_id`` to the run id, and the caller cannot control it — so no
        payload field can redirect verification to a different job. It then
        resolves that job's PR, reads the PR's head branch from GitHub, and
        polls the PR head's CI (reusing the workflows ``ci_green`` gate
        machinery).

        This method absorbs the wait for the coding run to finish (#2303,
        sixth pass). The ``talon_run`` stage dispatches ``wait: false`` and
        returns within seconds; the workflows runner has **no** primitive that
        parks a ``signal_status_ok`` stage on the ``talon:<job_id>`` waitable
        the dispatch hands back (only ``consent_collect`` / ``council_approve``
        gates induce WAITING), so the SEQUENTIAL edge fires ``verify_ci``
        immediately, while the (hour-plus) coding job is still running. This
        method therefore first does a fast point-in-time reap (settling a job
        that already finished — including one recovered across a restart from
        its exit sidecar), and if the job is still non-terminal it waits on the
        durable ``talon:<job_id>`` rail (the same rail ``wait('talon:...')``
        uses) up to ``max_job_wait_seconds`` before reading its final status.
        Only after the ceiling elapses without the job going terminal does it
        fail closed (the run can be re-verified) — it never claims CI state for
        an unfinished job.

        Every verification fact (repo, PR number, claim/iterate mode) is read
        from the resolved **job record** the coordinator itself established at
        dispatch — never from caller-influenceable payload. This categorically
        ends the untrusted-payload pattern: approval markers, branch, job id,
        and repo/issue correlation were all caller-influenceable and are all now
        ignored for binding (#2303, fourth-pass structural directive).

        Fail-closed: returns ``{"ci_green": False, "reason": ...}`` whenever the
        run has no bound job in the map, the job never reached a terminal state
        within ``max_job_wait_seconds``, the job did not COMPLETE, the PR/branch
        cannot be resolved, or CI is not observably green.
        """

        # 1. Resolve the talon job THIS run dispatched — EXCLUSIVELY from the
        # run_id→job_id map keyed by the workflow run's session_id. No
        # caller-supplied job_id, no (repo, issue|pr) correlation fallback: if
        # the map has no entry for this run_id, fail closed (#2303).
        effective_run_id = run_id or self._current_workflow_run_id()
        if not effective_run_id:
            return {
                "ci_green": False,
                "repo": self._resolve_repo(repo) if repo else repo,
                "reason": "no_workflow_run_id",
            }
        target_id = self._pipeline_run_jobs.get(effective_run_id)
        if not target_id or target_id not in self._jobs:
            return {
                "ci_green": False,
                "repo": self._resolve_repo(repo) if repo else repo,
                "run_id": effective_run_id,
                "reason": "no_run_bound_job",
            }

        # Every verification fact is read from the job record the coordinator
        # established at dispatch, never from caller payload.
        info = self._jobs.get(target_id) or {}
        repo_resolved = self._resolve_repo(info.get("repo") or repo or "")
        job_issue = info.get("issue")
        job_pr = info.get("pr")
        # A job dispatched in iterate mode records its target ``pr``; a claim
        # job records its ``issue``. Derive the mode from the job, not payload.
        job_mode = "iterate" if job_pr is not None else "claim"

        def _fail(reason: str, **extra: Any) -> Dict[str, Any]:
            return {
                "ci_green": False,
                "repo": repo_resolved,
                "issue": job_issue,
                "pr": job_pr,
                "job_id": target_id,
                "reason": reason,
                **extra,
            }

        from kestrel_sovereign.signals.sources.talon_pipeline import (
            _await_completion,
            _find_pr_url,
        )

        # 2. Ensure the talon job is terminal before verifying its PR's CI
        # (#2303, sixth pass). Fast path: a point-in-time reap settles a job
        # that already finished (reaping its process handle / exit sidecar —
        # this also recovers a job whose process finished across a restart).
        if self._reap_cli_job(info):
            self._persist_jobs()
        final_status = info.get("status")

        if final_status not in _TERMINAL_TALON_STATES:
            # Still running. The ``talon_run`` stage dispatched ``wait: false``
            # and the workflows runner has no primitive that parks a
            # ``signal_status_ok`` stage on the ``talon:<job_id>`` waitable, so
            # ``verify_ci`` fires seconds after dispatch while the (hour-plus)
            # coding job is still going. Absorb the wait HERE, on the durable
            # rail the dispatch handed back — the same one ``wait('talon:...')``
            # drives — up to the held-turn ceiling. Failing closed instead
            # would make the pipeline never go green for any real run.
            try:
                await _await_completion(self, target_id, max_job_wait_seconds)
            except Exception as exc:  # noqa: BLE001 - fail closed on wait error
                return _fail(f"job_wait_error:{exc}", job_id=target_id)
            # The wait rail reaps + persists in place (and may reload the job
            # dict across a restart), so re-read the record for its final state.
            info = self._jobs.get(target_id) or info
            final_status = info.get("status")

        if final_status not in _TERMINAL_TALON_STATES:
            # The job outlasted the ceiling. Fail closed — the run can be
            # re-verified — rather than claim CI state for an unfinished job.
            return _fail(
                f"job_still_running:{final_status or 'unknown'}",
                job_id=target_id,
                detail=(
                    f"talon job did not reach a terminal state within "
                    f"{max_job_wait_seconds}s; watch it via "
                    f"wait('talon:{target_id}') (fail closed)"
                ),
            )

        # 2b. Require the talon job to have SUCCEEDED before polling CI. A job
        # that ended terminal-failed (``failed`` / ``reject`` /
        # ``finished_unknown``) may still have opened a PR whose checks are
        # green; treating that as ``ci_green`` would greenlight a failed run.
        # Only ``complete`` proceeds to CI verification (#2303, fail closed).
        if final_status != _SUCCESSFUL_TALON_STATE:
            return _fail(
                f"job_not_complete:{final_status or 'unknown'}", job_id=target_id
            )

        # 3. Resolve the PR the talon job opened. In iterate mode the job record
        # already carries its target ``pr``; else parse it from the PR the job
        # opened. Both come from the job the coordinator dispatched, not payload.
        pr_url = _find_pr_url(self, target_id)
        target_pr = int(job_pr) if (job_mode == "iterate" and job_pr) else None
        if target_pr is None:
            target_pr = _pr_number_from_url(pr_url)
        if target_pr is None:
            return _fail("pr_not_found", job_id=target_id, pr_url=pr_url)

        # 4. Resolve the PR's head branch from GitHub — never a caller branch.
        head = self._github_pr_head(repo_resolved, target_pr)
        if not head or not head.get("ref"):
            return _fail("pr_head_unresolved", job_id=target_id, pr=target_pr)
        branch = str(head["ref"])

        # 5. Poll the PR head branch's CI, reusing the workflows ci_green gate.
        try:
            from kestrel_feature_workflows.models import Gate
            from kestrel_feature_workflows.runner import (
                _ci_marker_green,
                _default_ci_green_provider,
            )
        except Exception as exc:  # noqa: BLE001 - workflows feature is optional
            return _fail(
                f"ci_green_unavailable:{exc}",
                job_id=target_id,
                pr=target_pr,
                branch=branch,
            )

        gate_params: Dict[str, Any] = {
            "repo": repo_resolved,
            "branch": branch,
            "poll_interval_seconds": poll_interval_seconds,
            "max_wait_seconds": max_ci_wait_seconds,
        }
        if required_checks:
            gate_params["required_checks"] = list(required_checks)
        gate = Gate(type="ci_green", params=gate_params)
        marker = await _default_ci_green_provider(gate, None)
        green = _ci_marker_green(gate, marker)
        marker_status = marker.get("status") if isinstance(marker, dict) else None
        return {
            "ci_green": bool(green),
            "repo": repo_resolved,
            "issue": job_issue,
            "pr": target_pr,
            "pr_url": pr_url,
            "branch": branch,
            "head_sha": head.get("sha"),
            "job_id": target_id,
            "marker_status": marker_status,
            "reason": None if green else (marker_status or "ci_not_green"),
        }

    def _github_pr_head(self, repo: str, pr: int) -> Optional[Dict[str, Any]]:
        """Fetch a PR's head ``{ref, sha}`` from GitHub (read-only).

        Uses the same token env as the git/clone paths. Returns None on any
        error so the caller fails closed rather than trusting stale data.
        """
        token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_PAT")
        )
        if not token:
            return None
        url = f"https://api.github.com/repos/{repo.strip()}/pulls/{int(pr)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "kestrel-fleet-ci-probe",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
                doc = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - fail closed on any GitHub error
            return None
        head = doc.get("head") if isinstance(doc, dict) else None
        if not isinstance(head, dict):
            return None
        return {"ref": head.get("ref"), "sha": head.get("sha")}

    @tool(
        name="talon_file_and_claim",
        description=(
            "Loop-closing primitive: file a GitHub issue, then "
            "immediately dispatch it to Talon. Runs `gh issue create` "
            "through the audited computer_use shell path (auto-approved "
            "if the Sovereign added a matching allowlist pattern), parses "
            "the new issue number, then calls talon_claim. Returns the "
            "issue URL and the Talon job_id."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon file-and-claim",
    )
    async def talon_file_and_claim(
        self,
        title: str,
        body: str,
        labels: Optional[str] = None,
        repo: str = "KestrelSovereignAI/kestrel-sovereign",
    ) -> ToolResult:
        """File an issue and dispatch it to Talon in one hop.

        This is the single primitive that closes Emma's loop: instead of
        the Sovereign typing `gh issue create` and then a separate claim,
        the agent calls this once. The `gh` invocation goes through
        ``ComputerUseFeature.shell`` so the scoped auto-approve policy
        (epic #1290 / D1) governs it and a full audit row is written —
        the agent never shells out unaudited.

        Args:
            title: Issue title.
            body: Issue body (markdown).
            labels: Optional comma-separated label names. Talon's
                reserved lifecycle labels (``agent-claimed`` etc.) are
                dropped automatically — Talon applies ``agent-claimed``
                itself when it claims, and pre-stamping it would make the
                claim abort as "already claimed".
            repo: ``owner/name`` (default the sovereign repo; ``"self"``
                resolves the same way ``talon_claim`` does).

        Returns:
            ``ToolResult.ok`` with ``{filed, issue_url, issue_number,
            job_id, dispatched}``. ``failed`` if the issue could not be
            created or its number could not be parsed; ``partial`` if the
            issue was filed but the Talon dispatch did not take.
        """
        repo_resolved = self._resolve_repo(repo)

        agent = getattr(self, "agent", None)
        cu = (
            agent.get_feature("ComputerUseFeature")
            if agent is not None and hasattr(agent, "get_feature")
            else None
        )
        if cu is None:
            await self._log_fac_outcome(
                decision="missing_computer_use",
                reason_code="MISSING_COMPUTER_USE",
                repo=repo_resolved,
                filed=False,
                dispatched=False,
            )
            return ToolResult.failed(
                "talon_file_and_claim: MISSING_COMPUTER_USE — "
                "ComputerUseFeature unavailable, refusing to file the "
                "issue via an unaudited shell. Enable computer_use so the "
                "scoped auto-approve policy and audit row apply.",
                data={
                    "filed": False,
                    "dispatched": False,
                    "reason_code": "MISSING_COMPUTER_USE",
                    "repo": repo_resolved,
                },
            )

        # Drop Talon's reserved lifecycle labels (esp. ``agent-claimed``).
        # Filing the issue with ``agent-claimed`` makes the very next
        # ``talon_claim`` abort — Talon's ``is_claimed()`` sees the label
        # and reports "Issue #N is already claimed", so the loop never
        # closes (root cause of the #1299/#1301/#1303 Talon failures).
        # Talon stamps ``agent-claimed`` itself as part of claiming.
        requested = [
            label.strip()
            for label in (labels or "").split(",")
            if label.strip()
        ]
        applied_labels = [
            label
            for label in requested
            if label.lower() not in _TALON_RESERVED_LABELS
        ]
        stripped_labels = [
            label for label in requested if label.lower() in _TALON_RESERVED_LABELS
        ]

        async def _run_create(cmd_labels: List[str]):
            parts = [
                "gh", "issue", "create",
                "-R", repo_resolved,
                "--title", title,
                "--body", body,
            ]
            for lab in cmd_labels:
                parts += ["--label", lab]
            res = await cu.shell(shlex.join(parts), timeout=120)
            d = res.data or {}
            out = str(d.get("stdout", ""))
            err = str(d.get("stderr", ""))
            ok = res.status == "ok"
            mm = re.search(r"https://github\.com/\S+/issues/(\d+)", out)
            return res, ok, out, err, mm

        shell_res, succeeded, stdout, stderr, m = await _run_create(
            applied_labels
        )

        # A loop-closing primitive must not die on a bad label. ``gh
        # issue create`` hard-fails (exit 1, no issue) if ANY ``--label``
        # does not already exist in the repo — one typo'd label from the
        # LLM would otherwise sink the whole file+claim. If the create
        # failed while we passed labels and the error is label-related,
        # retry ONCE with no labels: the issue (the point) still gets
        # filed, Talon stamps ``agent-claimed`` itself, and the dropped
        # labels are reported (never silently). The retry is still
        # ``gh issue create -R <repo> …`` so it stays inside the scoped
        # auto-approve seed — no new human gate.
        dropped_unknown_labels: List[str] = []
        label_retry = False
        if (not succeeded or m is None) and applied_labels:
            blob = (
                f"{stderr}\n{stdout}\n{shell_res.error or ''}"
            ).lower()
            if "label" in blob and (
                "not found" in blob
                or "could not add" in blob
                or "not a valid" in blob
                or "no label" in blob
            ):
                label_retry = True
                dropped_unknown_labels = list(applied_labels)
                applied_labels = []
                shell_res, succeeded, stdout, stderr, m = (
                    await _run_create([])
                )

        if not succeeded or m is None:
            reason_code, hint = _gh_failure_reason(
                shell_res, stdout, stderr,
                succeeded=succeeded, parsed_url=m is not None,
            )
            shell_data = shell_res.data or {}
            shell_returncode = shell_data.get("returncode")
            shell_timed_out = bool(shell_data.get("timed_out"))
            await self._log_fac_outcome(
                decision="filing_failed",
                reason_code=reason_code,
                repo=repo_resolved,
                filed=False,
                dispatched=False,
                extra={
                    "shell_status": str(shell_res.status),
                    "shell_returncode": shell_returncode,
                    "shell_timed_out": shell_timed_out,
                    "label_retry": label_retry,
                    "dropped_unknown_labels": dropped_unknown_labels,
                },
            )
            return ToolResult.failed(
                f"talon_file_and_claim: {reason_code} — {hint}",
                data={
                    "filed": False,
                    "dispatched": False,
                    "reason_code": reason_code,
                    "remediation": hint,
                    "repo": repo_resolved,
                    "shell_status": str(shell_res.status),
                    "shell_error": shell_res.error,
                    "shell_returncode": shell_returncode,
                    "shell_timed_out": shell_timed_out,
                    "stderr_tail": stderr[-300:],
                    "stdout_tail": stdout[-300:],
                    "label_retry": label_retry,
                    "dropped_unknown_labels": dropped_unknown_labels,
                },
            )

        issue_url = m.group(0)
        issue_number = int(m.group(1))

        claim = await self.talon_claim(repo=repo_resolved, issue=issue_number)
        claim_data = claim.data or {}
        dispatched = bool(claim_data.get("dispatched"))
        # Accept all three identifier shapes the dispatch paths produce:
        # CLI-background → ``job_id``; A2A (post-#1368) → ``task_id``;
        # historical mesh path (deleted) used ``message_id`` — left in
        # the lookup chain only for any stale serialized state that
        # could still surface during the rollout window.
        job_id = (
            claim_data.get("job_id")
            or claim_data.get("task_id")
            or claim_data.get("message_id")
            or None
        )
        result_data = {
            "filed": True,
            "issue_url": issue_url,
            "issue_number": issue_number,
            "repo": repo_resolved,
            "dispatched": dispatched,
            "job_id": job_id,
            "applied_labels": applied_labels,
            "stripped_labels": stripped_labels,
            "dropped_unknown_labels": dropped_unknown_labels,
            "label_retry": label_retry,
            "claim": claim_data,
        }
        notes = []
        if stripped_labels:
            notes.append(
                f"dropped Talon-reserved label(s) {stripped_labels} so the "
                f"claim wasn't pre-empted"
            )
        if dropped_unknown_labels:
            notes.append(
                f"refiled without unknown/invalid label(s) "
                f"{dropped_unknown_labels} (not present in the repo) so the "
                f"loop still closed"
            )
        stripped_note = f" ({'; '.join(notes)})" if notes else ""
        if dispatched:
            await self._log_fac_outcome(
                decision="filed_and_dispatched",
                reason_code="OK",
                repo=repo_resolved,
                filed=True,
                dispatched=True,
                issue_number=issue_number,
                job_id=job_id,
            )
            return ToolResult.ok(
                confirmation=(
                    f"Filed {repo_resolved}#{issue_number} ({issue_url}) "
                    f"and dispatched it to Talon (job_id={job_id})"
                    f"{stripped_note}."
                ),
                data=result_data,
            )
        # Issue filed but dispatch didn't take — surface as PARTIAL with
        # a discriminated reason code so the operator can tell "issue is
        # live, please claim by hand" from "everything broke." This is
        # the asymmetric-outcome case the partial-only-when-both-layers-
        # attempted rule was written for.
        result_data["reason_code"] = "DISPATCH_FAILED"
        result_data["remediation"] = (
            "the GitHub issue exists; call talon_claim(repo, issue) "
            "manually or investigate why the dispatch path returned "
            "dispatched=False."
        )
        await self._log_fac_outcome(
            decision="filed_dispatch_failed",
            reason_code="DISPATCH_FAILED",
            repo=repo_resolved,
            filed=True,
            dispatched=False,
            issue_number=issue_number,
        )
        return ToolResult.partial(
            confirmation=(
                f"Filed {repo_resolved}#{issue_number} ({issue_url}) but "
                f"the Talon dispatch did not take."
            ),
            error=(
                "talon_file_and_claim: DISPATCH_FAILED — "
                + (claim.error or "talon_claim did not report dispatched")
            ),
            data=result_data,
        )

    # ------------------------------------------------------------------
    # Internal: outcome audit row
    # ------------------------------------------------------------------

    async def _log_fac_outcome(
        self,
        *,
        decision: str,
        reason_code: str,
        repo: str,
        filed: bool,
        dispatched: bool,
        issue_number: Optional[int] = None,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an after-execution outcome row to ``security_audit_log``.

        The gate's pre-execution row records only the DECISION
        (auto_mode_allowed). #1383 demands that the OUTCOME of the tool
        run be recorded too — without that, an operator scanning the
        audit log can see "allowed" but not "what actually happened."

        Writes are best-effort: a missing SecurityFeature must not
        crash the loop-closing primitive. We log warnings instead.
        """
        security = self._get_security_feature()
        store = getattr(security, "permission_store", None) if security else None
        if store is None or not hasattr(store, "log_decision"):
            logger.debug(
                "talon_file_and_claim outcome not audited "
                f"(SecurityFeature/permission_store unavailable): "
                f"decision={decision} reason={reason_code} repo={repo}"
            )
            return

        summary: Dict[str, Any] = {
            "reason_code": reason_code,
            "repo": repo,
            "filed": filed,
            "dispatched": dispatched,
        }
        if issue_number is not None:
            summary["issue_number"] = issue_number
        if job_id is not None:
            summary["job_id"] = job_id
        if extra:
            summary.update(extra)
        try:
            await store.log_decision(
                feature_name="talon_feature",
                tool_name="talon_file_and_claim.outcome",
                action="tool_outcome",
                decision=decision,
                user_choice=None,
                args_summary=json.dumps(summary, default=str)[:1000],
            )
        except Exception as e:  # noqa: BLE001 — never break the caller
            logger.warning(
                f"Failed to write talon_file_and_claim outcome audit "
                f"row (reason_code={reason_code}): {e}"
            )

    @tool(
        name="talon_github_write",
        description=(
            "Bounded GitHub issue-write job for orchestration (#2581 — the "
            "`github.write` / `issue.close` capability): close, reopen, "
            "comment on, label, or update a GitHub issue after work "
            "completes, closing the claim→work→close loop without a human "
            "running `gh`. The GitHub token is used in-process for a single "
            "authenticated REST call and is NEVER handed to a shell or the "
            "read-only git/verify surface. Write targets are restricted to "
            "the agent's own repo (GITHUB_SELF_REPO) plus GITHUB_FLEET_REPOS. "
            "operation ∈ {close_issue, reopen_issue, comment, add_labels, "
            "remove_labels, update_issue}."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon github-write",
    )
    async def talon_github_write(
        self,
        operation: str,
        issue: int,
        repo: str = "KestrelSovereignAI/kestrel-sovereign",
        body: Optional[str] = None,
        labels: Optional[str] = None,
        title: Optional[str] = None,
        state_reason: Optional[str] = None,
    ) -> ToolResult:
        """Perform a scoped, audited GitHub issue write.

        This is the orchestration-side write path the read-only execution
        surface lacked (#2581). Unlike ``talon_file_and_claim`` — which shells
        out to ``gh issue create`` — this uses a single in-process REST call,
        so the credential is never exposed to a shell or the untrusted
        git/verify surface. The target repo is authorized against the write
        allowlist (own repo + fleet) before any request is made, and the
        outcome is written to the security audit log.

        Args:
            operation: One of ``close_issue``, ``reopen_issue``, ``comment``,
                ``add_labels``, ``remove_labels``, ``update_issue``.
            issue: Issue number (``123``, ``"#123"``, and ``"123"`` accepted).
            repo: ``owner/name`` (``"self"`` resolves to ``GITHUB_SELF_REPO``).
                Must be the agent's own repo or a ``GITHUB_FLEET_REPOS`` entry.
            body: Comment text (``comment``) or new issue body (``update_issue``).
            labels: Comma-separated label names (``add_labels`` / ``remove_labels``).
            title: New issue title (``update_issue``).
            state_reason: ``completed`` (default) or ``not_planned``
                (``close_issue``).

        Returns:
            ``ToolResult.ok`` with ``{operation, repo, issue, results}`` on
            success; ``failed`` on a validation, auth, or GitHub API error.
        """
        repo_resolved = self._resolve_repo(repo)
        try:
            target_repo = resolve_write_repo(repo_resolved)
            issue_number = parse_issue_number(issue)
            requests = build_github_write_requests(
                operation,
                target_repo,
                issue_number,
                body=body,
                labels=labels,
                title=title,
                state_reason=state_reason,
            )
        except GithubWriteError as e:
            await self._log_github_write_outcome(
                operation=operation,
                repo=repo_resolved,
                issue=issue,
                ok=False,
                reason_code="INVALID_REQUEST",
                detail=str(e),
            )
            return ToolResult.failed(
                f"talon_github_write: INVALID_REQUEST — {e}",
                data={
                    "success": False,
                    "operation": operation,
                    "repo": repo_resolved,
                    "issue": issue,
                    "reason_code": "INVALID_REQUEST",
                    "error": str(e),
                },
            )

        token = (
            os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PAT")
        )
        if not token:
            await self._log_github_write_outcome(
                operation=operation,
                repo=target_repo,
                issue=issue_number,
                ok=False,
                reason_code="NO_TOKEN",
                detail="no GITHUB_TOKEN/GH_TOKEN/GITHUB_PAT in environment",
            )
            return ToolResult.failed(
                "talon_github_write: NO_TOKEN — set GITHUB_TOKEN, GH_TOKEN, or "
                "GITHUB_PAT in the kestrel-sovereign environment so the "
                "bounded job can authenticate to GitHub.",
                data={
                    "success": False,
                    "operation": operation,
                    "repo": target_repo,
                    "issue": issue_number,
                    "reason_code": "NO_TOKEN",
                },
            )

        results: List[Dict[str, Any]] = []
        ok = True
        for request in requests:
            outcome = await self._github_api_write(request, token)
            results.append(
                {
                    "summary": request.summary,
                    "method": request.method,
                    "status": outcome.get("status"),
                    "ok": outcome.get("ok"),
                    "error": outcome.get("error"),
                }
            )
            if not outcome.get("ok"):
                ok = False
                break

        summary_text = "; ".join(r["summary"] for r in results)
        if ok:
            await self._log_github_write_outcome(
                operation=operation,
                repo=target_repo,
                issue=issue_number,
                ok=True,
                reason_code="OK",
                detail=summary_text,
            )
            return ToolResult.ok(
                confirmation=(
                    f"{operation} on {target_repo}#{issue_number} succeeded "
                    f"({summary_text})."
                ),
                data={
                    "success": True,
                    "operation": operation,
                    "repo": target_repo,
                    "issue": issue_number,
                    "results": results,
                },
            )

        failed = next((r for r in results if not r.get("ok")), results[-1])
        await self._log_github_write_outcome(
            operation=operation,
            repo=target_repo,
            issue=issue_number,
            ok=False,
            reason_code="GITHUB_API_ERROR",
            detail=str(failed.get("error")),
        )
        return ToolResult.failed(
            f"talon_github_write: GITHUB_API_ERROR — {operation} on "
            f"{target_repo}#{issue_number} failed "
            f"(HTTP {failed.get('status')}): {failed.get('error')}",
            data={
                "success": False,
                "operation": operation,
                "repo": target_repo,
                "issue": issue_number,
                "reason_code": "GITHUB_API_ERROR",
                "results": results,
            },
        )

    async def _github_api_write(
        self, request, token: str, *, timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute one GitHub write request in-process.

        The token is used only for this authenticated REST call and never
        leaves the process — no subprocess, no shell, no environment
        inheritance. Runs the blocking ``urllib`` call in a worker thread so
        the event loop is not blocked.
        """
        return await asyncio.to_thread(
            self._github_api_write_sync, request, token, timeout
        )

    @staticmethod
    def _github_api_write_sync(
        request, token: str, timeout: int
    ) -> Dict[str, Any]:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "kestrel-talon-github-write",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = None
        if request.payload is not None:
            data = json.dumps(request.payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        http_request = urllib.request.Request(
            request.url, data=data, headers=headers, method=request.method
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed api.github.com host
                http_request, timeout=timeout
            ) as response:
                status = getattr(response, "status", None) or response.getcode()
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in request.success_statuses:
                return {"ok": True, "status": e.code, "response": None}
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - error body may be unreadable
                detail = ""
            message = extract_error_message(detail) or str(e)
            return {
                "ok": False,
                "status": e.code,
                "error": redact_secrets(message)[:500],
            }
        except Exception as e:  # noqa: BLE001 - network/URL errors fail closed
            return {
                "ok": False,
                "status": None,
                "error": redact_secrets(str(e))[:500],
            }

        parsed = None
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
        if status in request.success_statuses:
            return {"ok": True, "status": status, "response": parsed}
        return {
            "ok": False,
            "status": status,
            "error": redact_secrets(
                extract_error_message(raw) or f"HTTP {status}"
            )[:500],
        }

    async def _log_github_write_outcome(
        self,
        *,
        operation: str,
        repo: str,
        issue: Any,
        ok: bool,
        reason_code: str,
        detail: str = "",
    ) -> None:
        """Append a GitHub-write outcome row to ``security_audit_log``.

        Best-effort: a missing SecurityFeature must not crash the bounded
        write — we log a warning instead so the mutation still returns.
        """
        security = self._get_security_feature()
        store = getattr(security, "permission_store", None) if security else None
        if store is None or not hasattr(store, "log_decision"):
            logger.debug(
                "talon_github_write outcome not audited "
                "(SecurityFeature/permission_store unavailable): "
                "operation=%s repo=%s issue=%s ok=%s reason=%s",
                operation,
                repo,
                issue,
                ok,
                reason_code,
            )
            return
        summary = {
            "operation": operation,
            "repo": repo,
            "issue": issue,
            "ok": ok,
            "reason_code": reason_code,
            "detail": detail,
        }
        try:
            await store.log_decision(
                feature_name="talon_feature",
                tool_name="talon_github_write.outcome",
                action="tool_outcome",
                decision="github_write_ok" if ok else "github_write_failed",
                user_choice=None,
                args_summary=json.dumps(summary, default=str)[:1000],
            )
        except Exception as e:  # noqa: BLE001 — never break the caller
            logger.warning(
                "Failed to write talon_github_write outcome audit row "
                "(reason_code=%s): %s",
                reason_code,
                e,
            )

    @tool(
        name="talon_get_config",
        description=(
            "Read Talon runtime policy and mutable preference. This is the "
            "agent's control surface for its coding-agent backend/model, "
            "separate from normal chat LLM routing."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon config",
    )
    async def talon_get_config(self) -> ToolResult:
        """Return effective Talon policy and preference."""
        try:
            policy, preference = load_talon_policy_preference()
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "error": str(e)},
            )
        return ToolResult.ok(
            confirmation=(
                "Talon config loaded "
                f"(default={preference.default_backend}/{preference.default_model})"
            ),
            data={
                "success": True,
                "policy": asdict(policy),
                "preference": asdict(preference),
            },
        )

    @tool(
        name="talon_set_config",
        description=(
            "Update mutable Talon preferences only (operator policy is not "
            "changed by this tool). Writes the same defaults that talon_claim "
            "consumes per-dispatch. Allowed values: "
            "default_backend ∈ {claude, codex, opencode}; "
            "default_model — when backend=claude one of {opus, sonnet, haiku} "
            "(required), when backend=codex/opencode optional (omit to use the "
            "provider default; an explicitly blank value is rejected); "
            "default_auth_lane ∈ {oauth, api_key, provider_config}. "
            "Cross-field rules: codex ⇒ auth_lane=oauth; "
            "opencode ⇒ auth_lane=provider_config; "
            "claude ⇒ auth_lane oauth or api_key. "
            "max_iterations / max_turns are positive integer counts."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon set-config",
    )
    async def talon_set_config(
        self,
        default_backend: Optional[str] = None,
        default_model: Optional[str] = None,
        default_auth_lane: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_turns: Optional[int] = None,
        skip_clarification: Optional[bool] = None,
        self_review: Optional[bool] = None,
    ) -> ToolResult:
        """Persist Talon preference updates under ``[talon.preference]``.

        Only the fields you pass are changed; ``None`` leaves the existing
        value untouched. These are the same controls talon_claim accepts
        per-dispatch — set them here to change the defaults. Operator policy
        (allowed_backends, billing, worktree requirements) is NOT writable
        here.

        Args:
            default_backend: Talon runtime backend — one of ``claude``, ``codex``, or ``opencode``. Separate from Kestrel chat LLM routing.
            default_model: Backend-specific model. When backend is ``claude``, one of ``opus``, ``sonnet``, or ``haiku`` (required). When backend is ``codex`` or ``opencode``, OPTIONAL — omit/leave unset to use the provider default; if you do pass a value it must be non-blank (an explicitly empty value is rejected).
            default_auth_lane: One of ``oauth``, ``api_key``, or ``provider_config``. Cross-field rules enforced downstream: ``codex`` requires ``oauth``; ``opencode`` requires ``provider_config``; ``claude`` accepts ``oauth`` or ``api_key`` (``api_key`` also requires policy ``allow_api_billing``).
            max_iterations: Default max LLM implementation iterations — a positive integer count (>= 1).
            max_turns: Default max agent turns per Talon iteration — a positive integer count (>= 1).
            skip_clarification: Default for skipping the analysis/clarification phase.
            self_review: Default for running Talon's self-review pass.

        Returns:
            ``{"success": True, ...}`` with the persisted preference on
            success; ``{"success": False, "error": ...}`` on a validation
            or write failure.
        """
        updates = {
            "default_backend": default_backend,
            "default_model": default_model,
            "default_auth_lane": default_auth_lane,
            "max_iterations": max_iterations,
            "max_turns": max_turns,
            "skip_clarification": skip_clarification,
            "self_review": self_review,
        }
        try:
            result = write_talon_preference(updates)
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "error": str(e)},
            )
        return ToolResult.ok(
            confirmation="Talon preference updated",
            data={"success": True, **result},
        )

    @tool(
        name="talon_verify",
        description=(
            "Reviewer-side audited test verification. Run one or more "
            "test commands and report a precise result state per command "
            "(passed / failed / blocked_by_policy / blocked_by_user / "
            "blocked_by_sandbox / tooling_error). Allowlisted project test "
            "commands (e.g. `uv run pytest ...`) run without prompting; "
            "anything else is approval-gated. Use this instead of ad-hoc "
            "shell so test evidence is structured and audited, and so a "
            "sandbox/policy block is never mislabeled as a user denial."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon verify",
    )
    async def talon_verify(
        self,
        commands: str,
        repo: str = "self",
        cwd: Optional[str] = None,
        ref: str = "",
        timeout: int = 600,
        note: str = "",
    ) -> ToolResult:
        """Run targeted verification commands and report structured evidence.

        Args:
            commands: One command per line (blank lines ignored).
                Allowlisted test runners run directly; non-allowlisted
                commands require approval and the block reason is recorded
                precisely.
            repo: ``owner/name`` (or ``self``) — used to locate the
                workspace clone to run in when ``cwd`` is not given.
            cwd: Working directory override. Must be a sandboxed workspace
                clone — never the running agent's source tree. Defaults to
                the repo's provisioned workspace clone; if none exists the
                call refuses with a structured
                ``workspace_not_provisioned`` result pointing to
                ``talon_setup_workspace(repo)``.
            ref: PR number / branch name / commit SHA to verify. When
                given, the workspace clone is fetched and checked out to
                that ref BEFORE any command runs, so a PR is verified
                against the PR's code and not whatever happened to be
                checked out (e.g. ``main``). A PR number may be written
                ``1630``, ``#1630``, or ``pr/1630``. If the fetch/checkout
                fails, no command runs and the result state is
                ``tooling_error`` — never a misleading test failure
                against the un-switched tree (issue #1631).
            timeout: Per-command wall-clock cap in seconds.
            note: Optional reviewer note included in the evidence (e.g.
                "local tests could not run; CI is the remaining hard gate").

        Returns:
            ``ToolResult.ok`` when every command passed; ``partial`` when
            any command failed, was blocked, or could not run (the tool
            itself ran fine — the LLM must not claim a clean pass).
            ``data`` carries the full ``VerificationEvidence`` dict plus
            ``repo``, ``cwd``, ``requested_ref``, ``checked_out_ref`` and
            ``head_sha`` so a reviewer can tell exactly which ref was
            verified, and ``confirmation`` carries the markdown evidence
            block for review/merge notes.
        """
        command_list = [c.strip() for c in str(commands).splitlines() if c.strip()]
        if not command_list:
            return ToolResult.failed(
                "talon_verify: no commands provided (one per line).",
                data={"success": False, "overall_state": "not_run"},
            )

        try:
            run_cwd = self._resolve_verify_cwd(repo, cwd)
        except _VerifyCwdError as e:
            return ToolResult.failed(str(e), data=e.data)

        requested_ref = (ref or "").strip()
        if requested_ref:
            checkout = await self._git_checkout_ref(run_cwd, requested_ref)
            if not checkout.get("ok"):
                # Fetch/checkout failed: refuse to run the requested
                # commands against the un-switched tree. Reporting those
                # commands as ``tooling_error`` (not ``failed``) is the
                # whole point of #1631 — a reviewer must be able to tell a
                # ref-selection failure from a real code failure.
                reason = checkout.get("error") or (
                    f"could not check out ref {requested_ref!r}"
                )
                summary = (
                    f"requested ref {requested_ref!r} could not be checked out "
                    f"in {run_cwd}: {reason}. Commands were NOT run; this is "
                    "a tooling/ref-selection failure, not a code failure."
                )
                evidence = VerificationEvidence(
                    results=[
                        TestCommandResult(
                            command=cmd,
                            state=VerificationState.TOOLING_ERROR,
                            allowlisted=False,
                            summary=summary,
                        )
                        for cmd in command_list
                    ],
                    note=note,
                )
                data = {
                    "success": True,
                    "repo": self._resolve_repo(repo),
                    "cwd": str(run_cwd),
                    "requested_ref": requested_ref,
                    "checked_out_ref": None,
                    "head_sha": None,
                    **evidence.to_dict(),
                }
                return ToolResult.partial(
                    evidence.to_markdown(),
                    f"verification overall state: {evidence.overall_state.value}",
                    data=data,
                )

        verifier = TalonVerifier(
            execute=self._make_verify_executor(run_cwd),
            approve=self._make_verify_approver(repo, run_cwd),
        )
        try:
            timeout_int = max(1, int(timeout))
        except (TypeError, ValueError):
            timeout_int = 600

        evidence = await verifier.verify_commands(
            command_list, timeout=timeout_int, note=note
        )
        head = await self._git_describe_head(run_cwd)
        data = {
            "success": True,
            "repo": self._resolve_repo(repo),
            "cwd": str(run_cwd),
            "requested_ref": requested_ref or None,
            "checked_out_ref": head.get("ref"),
            "head_sha": head.get("head_sha"),
            **evidence.to_dict(),
        }
        confirmation = evidence.to_markdown()
        if evidence.all_passed:
            return ToolResult.ok(confirmation=confirmation, data=data)
        return ToolResult.partial(
            confirmation,
            f"verification overall state: {evidence.overall_state.value}",
            data=data,
        )

    def _resolve_verify_cwd(self, repo: str, cwd: Optional[str]) -> Path:
        """Pick the directory to run verification commands in.

        Verification executes untrusted checked-out code (conftest.py, PR
        diffs), so the resolved directory must be a sandboxed workspace
        clone — never the running agent's own source tree (F301). Both the
        explicit ``cwd`` override and the workspace clone are run through
        :meth:`_assert_workspace_safe`. When no ``cwd`` is given and no
        workspace has been provisioned, this raises :class:`_VerifyCwdError`
        with a structured ``workspace_not_provisioned`` refusal pointing to
        ``talon_setup_workspace(repo)`` — there is deliberately no silent
        ``project_dir()`` fallback.
        """
        repo_resolved = self._resolve_repo(repo)
        if cwd:
            path = Path(cwd).expanduser().resolve()
            if not path.is_dir():
                raise _VerifyCwdError(
                    f"talon_verify: cwd is not a directory: {path}",
                    {"success": False, "overall_state": "not_run"},
                )
            unsafe_reason = self._assert_workspace_safe(path)
            if unsafe_reason:
                raise _VerifyCwdError(
                    unsafe_reason,
                    {
                        "success": False,
                        "overall_state": "not_run",
                        "state": "unsafe_workspace",
                        "error": unsafe_reason,
                    },
                )
            return path

        workspace = self._workspace_path_for(repo_resolved)
        unsafe_reason = self._assert_workspace_safe(workspace)
        if unsafe_reason:
            raise _VerifyCwdError(
                unsafe_reason,
                {
                    "success": False,
                    "overall_state": "not_run",
                    "state": "unsafe_workspace",
                    "error": unsafe_reason,
                },
            )
        if (workspace / ".git").exists():
            return workspace

        err_msg = (
            f"No talon workspace exists for {repo_resolved} at {workspace}. "
            "talon_verify will not run untrusted checked-out code in the "
            "running agent's source tree. Call talon_setup_workspace(repo) "
            "to provision a sandboxed clone, then retry."
        )
        raise _VerifyCwdError(
            err_msg,
            {
                "success": False,
                "overall_state": "not_run",
                "state": "workspace_not_provisioned",
                "error": err_msg,
                "workspace": self._workspace_state(repo_resolved),
                "next_step": f"talon_setup_workspace(repo='{repo_resolved}')",
            },
        )

    def _make_verify_executor(self, run_cwd: Path):
        """Build an async executor that runs a command in ``run_cwd``.

        Returns a :class:`CommandExecution` whose ``ran`` flag records
        whether the process actually executed, so the verifier can tell
        a real exit code from a tooling/sandbox failure.
        """

        async def _execute(command: str, *, timeout: int = 600) -> CommandExecution:
            try:
                argv = shlex.split(command)
            except ValueError as e:
                return CommandExecution(ran=False, error=f"unparseable command: {e}")
            if not argv:
                return CommandExecution(ran=False, error="empty command")

            # F302: the checked-out tree is untrusted (conftest.py runs at
            # collection, PR diffs run under test). Strip every provider/LLM
            # credential and the GitHub token so a malicious repo cannot
            # exfiltrate secrets or act as the agent on GitHub.
            env, _stripped = sanitize_untrusted_env()
            started = time.monotonic()
            try:
                result = await run_bounded_subprocess(
                    argv,
                    cwd=run_cwd,
                    env=env,
                    timeout=timeout,
                    max_output_bytes=_TALON_CAPTURE_LIMIT_BYTES,
                )
            except FileNotFoundError as e:
                return CommandExecution(
                    ran=False, error=f"command not found: {argv[0]} ({e})"
                )
            except PermissionError as e:
                return CommandExecution(
                    ran=False,
                    sandbox_denied=True,
                    error=f"sandbox refused to execute {argv[0]}: {e}",
                )
            except Exception as e:  # noqa: BLE001
                return CommandExecution(
                    ran=False, error=f"subprocess tooling error: {e}"
                )

            if result.timed_out:
                duration_ms = int((time.monotonic() - started) * 1000)
                return CommandExecution(
                    ran=False,
                    duration_ms=duration_ms,
                    error=f"command timed out after {timeout}s",
                )
            duration_ms = int((time.monotonic() - started) * 1000)
            return CommandExecution(
                ran=True,
                returncode=result.returncode,
                stdout=redact_secrets(result.stdout.decode(errors="replace")),
                stderr=redact_secrets(result.stderr.decode(errors="replace")),
                duration_ms=duration_ms,
            )

        return _execute

    def _make_verify_approver(self, repo: str, run_cwd: Path):
        """Build an async approver for non-allowlisted verify commands.

        Returns ``None`` (fail-closed, reported as ``blocked_by_policy``)
        when no SecurityFeature/approval queue is available. An explicit
        operator DENY is reported by the approval queue as a non-user
        scope, so :func:`verification.classify_denial` never mislabels it
        as a user denial.
        """

        async def _approve(command: str) -> Optional[tuple[bool, str]]:
            security = self._get_security_feature()
            queue = getattr(security, "approval_queue", None) if security else None
            if queue is None:
                return None
            try:
                approved, scope = await queue.request_approval(
                    feature_name="talon",
                    tool_name="verify_command",
                    tool_args={
                        "command": command,
                        "repo": self._resolve_repo(repo),
                        "cwd": str(run_cwd),
                    },
                )
                return bool(approved), str(scope)
            except (TimeoutError, asyncio.TimeoutError):
                return False, "timeout"
            except Exception as e:  # noqa: BLE001
                logger.error(f"talon_verify approval failed: {e}", exc_info=True)
                return None

        return _approve

    @staticmethod
    def _parse_verify_ref(ref: str) -> tuple[str, str]:
        """Classify a verify ref into ``("pr", number)`` or ``("ref", value)``.

        A PR number may be written ``1630``, ``#1630``, or ``pr/1630``
        (case-insensitive, with ``/``, ``#`` or ``-`` as the separator).
        Everything else — a branch name or a commit SHA — is returned as a
        plain ``ref`` and resolved by ``git checkout`` directly.
        """
        r = (ref or "").strip()
        m = re.match(r"^(?:#|pr[/#-]?)(\d+)$", r, re.IGNORECASE)
        if m:
            return "pr", m.group(1)
        if r.isdigit():
            return "pr", r
        return "ref", r

    async def _git_checkout_ref(self, workspace: Path, ref: str) -> Dict[str, Any]:
        """Fetch and check out ``ref`` in the workspace clone.

        ``ref`` forms (see :meth:`_parse_verify_ref`):

          * a PR number — fetched from ``refs/pull/<n>/head`` and checked
            out detached, so a PR is verified against its own head commit.
          * a branch name — the remote is fetched, then ``git checkout``
            switches to (and, if needed, creates a local tracking branch
            for) it.
          * a commit SHA — fetched best-effort, then checked out detached.

        Returns ``{"ok": bool, "error": str, "checked_out_ref": str,
        "head_sha": str}``. A non-``ok`` result means the requested ref
        could not be materialised; the caller must NOT run verification
        against the un-switched tree (issue #1631).
        """
        if not (workspace / ".git").exists():
            return {
                "ok": False,
                "error": (
                    f"{workspace} is not a git checkout; cannot select a "
                    "ref to verify"
                ),
            }
        kind, value = self._parse_verify_ref(ref)
        if not value:
            return {"ok": False, "error": "empty ref"}
        network_env = self._build_git_subprocess_env(require_github_token=False)

        if kind == "pr":
            fetch = await self._git_run(
                ["fetch", "origin", f"refs/pull/{value}/head"],
                cwd=workspace,
                timeout=120,
                env=network_env,
            )
            if not fetch.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git fetch of PR #{value} failed: "
                        f"{fetch.get('error') or 'unknown error'}"
                    ),
                }
            checkout = await self._git_run(
                ["checkout", "--force", "--detach", "FETCH_HEAD"],
                cwd=workspace,
                timeout=60,
            )
            if not checkout.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git checkout of PR #{value} failed: "
                        f"{checkout.get('error') or 'unknown error'}"
                    ),
                }
        else:
            # Branch or SHA. Fetch the remote so newly-pushed branches
            # and commits are available locally. If a remote branch exists,
            # make origin/<branch> the source of truth so a stale local
            # branch cannot be verified accidentally.
            fetch = await self._git_run(
                ["fetch", "--all", "--prune"],
                cwd=workspace,
                timeout=120,
                env=network_env,
            )
            if not fetch.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"git fetch before checking out ref {value!r} failed: "
                        f"{fetch.get('error') or 'unknown error'}"
                    ),
                }
            remote_branch = await self._git_run(
                ["rev-parse", "--verify", f"refs/remotes/origin/{value}"],
                cwd=workspace,
                timeout=30,
            )
            if remote_branch.get("ok"):
                checkout = await self._git_run(
                    ["checkout", "--force", "-B", value, f"origin/{value}"],
                    cwd=workspace,
                    timeout=60,
                )
                if not checkout.get("ok"):
                    return {
                        "ok": False,
                        "error": (
                            f"git checkout of remote branch {value!r} failed: "
                            f"{checkout.get('error') or 'unknown error'}"
                        ),
                    }
            else:
                checkout = await self._git_run(
                    ["checkout", "--force", value], cwd=workspace, timeout=60
                )
                if not checkout.get("ok"):
                    return {
                        "ok": False,
                        "error": (
                            f"git checkout of ref {value!r} failed: "
                            f"{checkout.get('error') or 'unknown error'}"
                        ),
                    }

        head = await self._git_describe_head(workspace)
        return {
            "ok": True,
            "checked_out_ref": head.get("ref"),
            "head_sha": head.get("head_sha"),
        }

    async def _git_run(
        self,
        args: List[str],
        *,
        cwd: Path,
        timeout: int = 120,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run a git subcommand in ``cwd`` and capture its outcome.

        The default git-only environment strips every provider, Kestrel, data,
        and GitHub credential. Network callers explicitly provide the narrower
        token-bearing environment from :meth:`_build_git_subprocess_env`;
        local commands never need that authority. Returns
        ``{"ok", "error", "stdout"}``.
        """
        if env is None:
            env = sanitize_untrusted_env()[0]
        try:
            # A workspace may be attacker-controlled. Override hooks for every
            # coordinator-owned git command so a planted post-checkout hook
            # cannot inherit even the narrowly-scoped GitHub credential.
            with tempfile.TemporaryDirectory(prefix="kestrel-git-hooks-") as hooks:
                result = await run_bounded_subprocess(
                    ["git", "-c", f"core.hooksPath={hooks}", *args],
                    cwd=cwd,
                    env=env,
                    timeout=timeout,
                    max_output_bytes=_TALON_CAPTURE_LIMIT_BYTES,
                )
        except FileNotFoundError as e:
            return {"ok": False, "error": f"git not found: {e}"}
        except (OSError, SubprocessCleanupError) as e:
            return {
                "ok": False,
                "error": redact_secrets(f"git could not run: {e}"),
            }
        except Exception as e:  # noqa: BLE001
            # Preserve the historical mapping-shaped private API even if the
            # shared runner itself encounters an unexpected collection error.
            # Cancellation remains transparent because CancelledError is not
            # an Exception on supported Python versions.
            return {
                "ok": False,
                "error": redact_secrets(f"git subprocess tooling error: {e}"),
            }
        if result.timed_out:
            return {
                "ok": False,
                "error": f"git {args[0]} timed out after {timeout}s",
            }
        stdout = redact_secrets(result.stdout.decode(errors="replace"))
        stderr = redact_secrets(result.stderr.decode(errors="replace"))
        if result.returncode != 0:
            return {
                "ok": False,
                "error": stderr[-500:]
                or f"git {args[0]} failed (exit {result.returncode})",
            }
        return {"ok": True, "stdout": stdout}

    @staticmethod
    def _build_git_subprocess_env(*, require_github_token: bool) -> Dict[str, str]:
        """Build a git-only env with no LLM, Kestrel, or data credentials."""

        env, _stripped = sanitize_untrusted_env()
        token = (
            os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PAT")
        )
        if require_github_token and not token:
            raise RuntimeError(
                "kestrel-talon needs GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT "
                "in the kestrel-sovereign environment to access GitHub."
            )
        if token:
            env["GITHUB_TOKEN"] = token
            env["GH_TOKEN"] = token
        return env

    async def _git_describe_head(self, workspace: Path) -> Dict[str, Optional[str]]:
        """Best-effort current ref + full HEAD SHA of a checkout.

        ``ref`` is the symbolic branch name when on a branch, else the
        short SHA (detached HEAD). ``head_sha`` is the full HEAD commit
        SHA. Either may be ``None`` if git can't be queried.
        """

        async def _run(args: List[str]) -> Optional[str]:
            result = await self._git_run(args, cwd=workspace, timeout=10)
            if not result.get("ok"):
                return None
            out = str(result.get("stdout") or "").strip()
            return out or None

        head_sha = await _run(["rev-parse", "HEAD"])
        ref = await _run(["rev-parse", "--abbrev-ref", "HEAD"])
        if ref == "HEAD":  # detached
            ref = await _run(["rev-parse", "--short", "HEAD"])
        return {"ref": ref, "head_sha": head_sha}

    @tool(
        name="talon_schedule_work_rescue",
        description=(
            "Schedule the stalled_work_rescue workflow as a SAFE recurring "
            "loop. Each tick detects stalled fleet work and requests fresh "
            "per-run consent; the irreversible dispatch/close stages never "
            "auto-run off the schedule. Refuses to bake in repair targets, "
            "resolution evidence, or a blanket approval marker."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon schedule-rescue",
    )
    async def talon_schedule_work_rescue(
        self,
        cron: Optional[str] = None,
        stale_days: int = 3,
    ) -> ToolResult:
        """Install a safe recurring ``stalled_work_rescue`` schedule (#2200).

        The scheduled task is the workflows feature's ``workflow_run`` tool,
        started against the built-in ``stalled_work_rescue`` definition with
        observation-only params. Detect and govern recur; dispatch and close
        only execute with explicit per-run approval and evidence, so the loop is
        safe to run unattended.

        Args:
            cron: Cron expression (5 fields) or alias. Defaults to the built-in
                cadence of every 6 hours.
            stale_days: How many idle days mark work as stalled (default 3).
        """
        from kestrel_sovereign.signals.sources.workflow_rescue import (
            UnsafeRecurringScheduleError,
            build_recurring_schedule_request,
        )

        try:
            request = build_recurring_schedule_request(
                cron=cron, stale_days=stale_days
            )
        except UnsafeRecurringScheduleError as exc:
            return ToolResult.failed(str(exc))

        agent = getattr(self, "agent", None)
        scheduler = (
            agent.get_feature("SchedulerFeature")
            if agent is not None and hasattr(agent, "get_feature")
            else None
        )
        if scheduler is None or not hasattr(scheduler, "schedule_add"):
            # No scheduler loaded — return the ready-to-use invocation so an
            # operator can install it by hand rather than silently no-op.
            return ToolResult.failed(
                "SchedulerFeature is not available; enable it to schedule the "
                "recurring rescue loop. Ready-to-use schedule_add args are in "
                "data['schedule_request'].",
                data={"schedule_request": request},
            )

        result = await scheduler.schedule_add(
            cron_expression=request["cron_expression"],
            task_name=request["task_name"],
            args_json=request["args_json"],
        )
        # Surface the request shape alongside whatever the scheduler returned.
        data = dict(getattr(result, "data", None) or {})
        data["schedule_request"] = request
        if getattr(result, "error", None):
            return ToolResult.failed(result.error, data=data)
        return ToolResult.ok(
            "Scheduled a safe recurring stalled_work_rescue loop "
            f"({request['cron_expression']}); dispatch/close still require "
            "fresh per-run approval and evidence.",
            data=data,
        )

    @tool(
        name="talon_workspace_status",
        description=(
            "Read-only: report on the talon workspace clone for a "
            "repo (path, exists, git HEAD, clean state, last fetch). "
            "No side effects."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon workspace-status",
    )
    async def talon_workspace_status(self, repo: str) -> ToolResult:
        """Inspect the talon workspace for ``repo``.

        Use this BEFORE ``talon_claim`` to verify the sandbox is
        ready: workspace exists, has a ``.git``, working tree is
        clean, last fetch was recent. If ``exists`` is False, call
        ``talon_setup_workspace`` first.
        """
        repo_resolved = self._resolve_repo(repo)
        state = await self._workspace_state_with_status(repo_resolved)
        data = {"success": True, "repo": repo_resolved, **state}

        # Honesty: this is a read-only inspect, but reporting OK on
        # an unprovisioned/unsafe workspace would let the agent
        # narrate "workspace ready" off a state that talon_claim will
        # immediately reject. Surface as PARTIAL with the failing
        # condition so the LLM has to speak it.
        if state.get("safe") is False:
            return ToolResult.partial(
                confirmation=f"Read workspace state for {repo_resolved}",
                error=state.get("unsafe_reason") or "workspace path is unsafe",
                data=data,
            )
        if not state.get("exists") or not state.get("is_git"):
            return ToolResult.partial(
                confirmation=f"Read workspace state for {repo_resolved}",
                error=(
                    f"workspace at {state.get('path')} is not provisioned "
                    "(exists=False or no .git); call talon_setup_workspace "
                    "before talon_claim"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Workspace ready: {repo_resolved} at {state.get('path')} "
                f"(head={state.get('head')}, clean={state.get('clean')})"
            ),
            data=data,
        )

    @tool(
        name="talon_setup_workspace",
        description=(
            "Provision (or refresh) a sandboxed talon workspace clone "
            "for a repo. The clone lives outside the running agent's "
            "source tree, so talon_claim can operate without ever "
            "touching the agent's own checkout. Approval-gated."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon setup-workspace",
    )
    async def talon_setup_workspace(
        self,
        repo: str,
        fetch: bool = True,
    ) -> ToolResult:
        """Clone ``repo`` into the canonical talon workspace path.

        If the workspace already exists as a git checkout and
        ``fetch`` is True, runs ``git fetch --all --prune`` to
        refresh remote refs. Approval-gated: this is a network +
        disk operation that creates persistent state under the
        agent's home directory.

        Args:
            repo: GitHub repo in ``owner/name`` format, or ``"self"``.
            fetch: If True (default), fetch remote refs after clone
                or against an existing clone.

        Returns:
            ``{success, state: "created" | "refreshed" | "exists",
              workspace: {...}}`` on success. Failure returns
            ``{success: False, error, ...}``.
        """
        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = self._assert_workspace_safe(workspace)
        if unsafe_reason:
            return ToolResult.failed(
                unsafe_reason,
                data={"success": False, "state": "unsafe_workspace", "error": unsafe_reason},
            )

        approved = await self._request_workspace_approval(
            repo=repo_resolved, workspace=workspace, fetch=fetch,
        )
        if not approved:
            return ToolResult.failed(
                "Workspace setup not approved",
                data={
                    "success": False,
                    "state": "approval_denied",
                    "error": "Workspace setup not approved",
                    "workspace": str(workspace),
                },
            )

        existing = self._workspace_state(repo_resolved)
        if existing["exists"] and existing["is_git"]:
            if fetch:
                fetch_result = await self._git_fetch(workspace)
                if not fetch_result["ok"]:
                    return ToolResult.failed(
                        fetch_result["error"],
                        data={
                            "success": False,
                            "state": "fetch_failed",
                            "error": fetch_result["error"],
                            "workspace": await self._workspace_state_with_status(
                                repo_resolved
                            ),
                        },
                    )
                return ToolResult.ok(
                    confirmation=f"Refreshed workspace for {repo_resolved} (git fetch)",
                    data={
                        "success": True,
                        "state": "refreshed",
                        "workspace": await self._workspace_state_with_status(
                            repo_resolved
                        ),
                    },
                )
            return ToolResult.ok(
                confirmation=f"Workspace already exists for {repo_resolved} (no fetch)",
                data={
                    "success": True,
                    "state": "exists",
                    "workspace": await self._workspace_state_with_status(
                        repo_resolved
                    ),
                },
            )

        # Need to clone. Make sure the parent dir exists.
        try:
            workspace.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.failed(
                str(e),
                data={"success": False, "state": "mkdir_failed", "error": str(e)},
            )

        clone_url = f"https://github.com/{repo_resolved}.git"
        clone_result = await self._git_clone(clone_url, workspace)
        if not clone_result["ok"]:
            return ToolResult.failed(
                clone_result["error"],
                data={
                    "success": False,
                    "state": "clone_failed",
                    "error": clone_result["error"],
                },
            )

        return ToolResult.ok(
            confirmation=f"Created workspace for {repo_resolved} at {workspace}",
            data={
                "success": True,
                "state": "created",
                "workspace": await self._workspace_state_with_status(repo_resolved),
            },
        )

    def _get_security_feature(self):
        agent = getattr(self, "agent", None)
        if agent is None:
            return None
        if hasattr(agent, "get_feature"):
            feat = agent.get_feature("SecurityFeature")
            if feat is not None:
                return feat
        features = getattr(agent, "features", None)
        if isinstance(features, dict):
            return features.get("SecurityFeature") or features.get("security")
        return None

    async def _request_workspace_approval(
        self, repo: str, workspace: Path, fetch: bool,
    ) -> bool:
        """Gate workspace creation/refresh through SecurityFeature.

        Setting up a workspace creates persistent state and pulls
        from the network — both worth a user-visible approval gate.
        Without an approval queue, dispatch is denied (fail-closed).
        """
        security = self._get_security_feature()
        if security is None or not hasattr(security, "approval_queue"):
            logger.warning(
                "SecurityFeature unavailable; talon_setup_workspace denied"
            )
            return False
        try:
            approved, _scope = await security.approval_queue.request_approval(
                feature_name="talon",
                tool_name="setup_workspace",
                tool_args={
                    "repo": repo,
                    "workspace_path": str(workspace),
                    "fetch": fetch,
                    "operation": (
                        "refresh existing clone"
                        if workspace.exists()
                        else "clone fresh"
                    ),
                },
            )
            return bool(approved)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except Exception as e:
            logger.error(f"Workspace approval failed: {e}", exc_info=True)
            return False

    async def _git_clone(self, url: str, dest: Path) -> Dict[str, Any]:
        try:
            env = self._build_git_subprocess_env(require_github_token=True)
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        result = await self._git_run(
            ["clone", url, str(dest)],
            cwd=dest.parent,
            timeout=300,
            env=env,
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "git clone failed"}
        return {"ok": True}

    async def _git_fetch(self, workspace: Path) -> Dict[str, Any]:
        try:
            env = self._build_git_subprocess_env(require_github_token=True)
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        result = await self._git_run(
            ["fetch", "--all", "--prune"],
            cwd=workspace,
            timeout=120,
            env=env,
        )
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or "git fetch failed"}
        return {"ok": True}

    @staticmethod
    def _resolve_repo(repo: str) -> str:
        if repo and repo.lower() == "self":
            return os.environ.get(
                "GITHUB_SELF_REPO", "KestrelSovereignAI/kestrel-sovereign"
            )
        return repo

    @staticmethod
    def _workspace_root() -> Path:
        """Where workspace clones live.

        Default: ``~/.kestrel/talon_workspaces/``. Override with
        ``KESTREL_TALON_WORKSPACE_ROOT``. Always OUTSIDE the running
        agent's source tree by construction.
        """
        override = os.environ.get("KESTREL_TALON_WORKSPACE_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return (Path.home() / ".kestrel" / "talon_workspaces").resolve()

    @classmethod
    def _workspace_path_for(cls, repo: str) -> Path:
        """Canonical workspace clone path for a given repo string.

        Always under ``_workspace_root()`` and never inside the
        running agent's source tree — so even if a user sets
        ``KESTREL_TALON_WORKSPACE_ROOT`` to a path that *would*
        overlap, ``_assert_workspace_safe`` rejects it before any
        dispatch.
        """
        owner_repo = repo.replace("/", "__")  # filesystem-safe
        return cls._workspace_root() / owner_repo

    @classmethod
    def _assert_workspace_safe(cls, workspace: Path) -> Optional[str]:
        """Return None if the workspace is safe to operate against, or
        an error string explaining why it isn't.

        "Safe" means the workspace path is NOT the running agent's
        source root and NOT a directory containing it. This is
        structural, not a flag the agent can bypass.
        """
        if _path_contains(workspace, _RUNNING_AGENT_SOURCE_ROOT):
            return (
                f"Workspace path {workspace} contains the running agent's "
                f"source tree {_RUNNING_AGENT_SOURCE_ROOT}. The dispatcher "
                "refuses to operate on the agent's own source — Talon "
                "would commit/branch/quality-check against the running "
                "process. Use code_edit / propose_improvement for "
                "constitutional self-modification, or set "
                "KESTREL_TALON_WORKSPACE_ROOT to a path outside the "
                "source tree."
            )
        if _path_contains(_RUNNING_AGENT_SOURCE_ROOT, workspace):
            return (
                f"Workspace path {workspace} is inside the running agent's "
                f"source tree {_RUNNING_AGENT_SOURCE_ROOT}. Move the "
                "workspace root outside the source tree (set "
                "KESTREL_TALON_WORKSPACE_ROOT)."
            )
        return None

    @classmethod
    def _workspace_state(cls, repo_resolved: str) -> Dict[str, Any]:
        """Read-only snapshot of a workspace clone's state.

        Returns a structural dict with: ``path``, ``exists``, ``is_git`` (has
        ``.git``), ``head`` (current ref or ``None``), ``clean`` (initially
        ``None``; :meth:`_workspace_state_with_status` fills it asynchronously),
        ``last_fetch_at`` (mtime of ``.git/FETCH_HEAD`` or ``None``),
        ``safe`` (passes ``_assert_workspace_safe``).
        """
        path = cls._workspace_path_for(repo_resolved)
        unsafe_reason = cls._assert_workspace_safe(path)

        state: Dict[str, Any] = {
            "repo": repo_resolved,
            "path": str(path),
            "exists": path.is_dir(),
            "is_git": False,
            "head": None,
            "clean": None,
            "last_fetch_at": None,
            "safe": unsafe_reason is None,
        }
        if unsafe_reason:
            state["unsafe_reason"] = unsafe_reason
            return state

        if not path.is_dir():
            return state

        git_dir = path / ".git"
        state["is_git"] = git_dir.exists()
        if not state["is_git"]:
            return state

        # HEAD ref
        head_file = git_dir / "HEAD"
        if head_file.is_file():
            try:
                head_text = head_file.read_text(encoding="utf-8").strip()
                if head_text.startswith("ref: refs/heads/"):
                    state["head"] = head_text[len("ref: refs/heads/"):]
                else:
                    state["head"] = head_text[:40]
            except OSError:
                pass

        fetch_head = git_dir / "FETCH_HEAD"
        if fetch_head.is_file():
            try:
                state["last_fetch_at"] = datetime.fromtimestamp(
                    fetch_head.stat().st_mtime, tz=timezone.utc,
                ).isoformat()
            except OSError:
                pass

        return state

    async def _workspace_state_with_status(self, repo_resolved: str) -> Dict[str, Any]:
        """Add bounded git cleanliness to the structural workspace snapshot."""

        state = dict(self._workspace_state(repo_resolved))
        if not state.get("safe") or not state.get("exists") or not state.get("is_git"):
            return state
        result = await self._git_run(
            ["status", "--porcelain"],
            cwd=Path(state["path"]),
            timeout=10,
            env=sanitize_untrusted_env()[0],
        )
        state["clean"] = (
            not str(result.get("stdout") or "").strip() if result.get("ok") else None
        )
        return state

    @tool(
        name="talon_batch",
        description=(
            "Dispatch a batch of issues to Talon from a PRD JSON file. "
            "``prd`` is REQUIRED and must be an absolute path to an "
            "existing PRD JSON file (there is no label/repo-scoped batch "
            "mode — kestrel-talon's `batch` subcommand only accepts "
            "`--prd`). Runs through the same policy layer as talon_claim: "
            "``allow_background_jobs`` and ``allowed_backends`` are "
            "enforced, the subprocess env is credential-sanitized, and a "
            "sandbox workspace clone must already be provisioned for "
            "``repo`` (call talon_setup_workspace first) — batch never "
            "operates on the running agent's source tree. Returns "
            "immediately with a job_id; poll talon_status."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon batch",
    )
    async def talon_batch(
        self,
        repo: str,
        prd: str = "",
    ) -> ToolResult:
        """Dispatch PRD batch processing to Talon.

        Like talon_claim, this launches in the background and returns
        immediately, and is guarded by the same Talon policy layer (F304):
        it loads ``[talon.policy]``/``[talon.preference]``, enforces
        ``allow_background_jobs``/``allowed_backends``, requires a
        provisioned sandbox workspace for ``repo``, and passes that
        workspace as ``--repo-dir`` plus a validated absolute PRD path.

        Args:
            repo: GitHub repo in owner/name format (or ``self``). Selects the sandbox workspace clone Talon runs the batch in (passed as ``--repo-dir``); the workspace must already be provisioned via talon_setup_workspace.
            prd: REQUIRED. Absolute path to an existing PRD JSON file. Relative paths and missing files are rejected before dispatch — pass an absolute path.

        Returns:
            ``{"dispatched": True, "job_id": ..., ...}`` on success — poll
            the returned ``job_id`` via ``talon_status`` (or
            ``talon_job_log``) to follow progress. Failure returns
            ``{"dispatched": False, "error": ...}``.
        """
        prd = (prd or "").strip()
        if not prd:
            return ToolResult.failed(
                "talon_batch requires an absolute prd path (PRD JSON file).",
                data={
                    "dispatched": False,
                    "state": "invalid_talon_runtime",
                    "error": "prd is required",
                },
            )

        try:
            policy, preference = load_talon_policy_preference()
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "invalid_talon_runtime",
                    "error": str(e),
                },
            )

        repo_resolved = self._resolve_repo(repo)
        workspace = self._workspace_path_for(repo_resolved)

        unsafe_reason = (
            self._assert_workspace_safe(workspace)
            if policy.require_sandboxed_workspace
            else None
        )
        if unsafe_reason:
            return ToolResult.failed(
                unsafe_reason,
                data={
                    "dispatched": False,
                    "state": "unsafe_workspace",
                    "error": unsafe_reason,
                },
            )

        # Refuse rather than fall through to the running source tree —
        # the same safeguard talon_claim uses.
        state = self._workspace_state(repo_resolved)
        if not state["exists"] or not state["is_git"]:
            err_msg = (
                "No talon workspace exists for "
                f"{repo_resolved} at {workspace}. Batch will not operate "
                "on the running agent's source tree. Call "
                "talon_setup_workspace(repo) to provision a sandboxed "
                "clone, then retry."
            )
            return ToolResult.failed(
                err_msg,
                data={
                    "dispatched": False,
                    "state": "workspace_not_provisioned",
                    "error": err_msg,
                    "workspace": state,
                    "next_step": (
                        f"talon_setup_workspace(repo='{repo_resolved}')"
                    ),
                },
            )

        prd_path = Path(prd).expanduser()
        try:
            execution = TalonBatchExecution(
                repo=repo_resolved,
                prd_path=prd_path,
                repo_dir=workspace,
            )
            invocation = build_talon_batch_invocation(
                TalonRuntimeRequest(),
                execution,
                policy=policy,
                preference=preference,
            )
        except TalonRuntimeError as e:
            return ToolResult.failed(
                str(e),
                data={
                    "dispatched": False,
                    "state": "talon_policy_rejected",
                    "error": str(e),
                },
            )

        cli_result = await self._dispatch_via_cli_background(
            invocation.argv,
            label=f"batch:{repo_resolved}:prd={prd_path.name}",
            env=invocation.env,
            extra_meta={
                "repo": repo_resolved,
                "prd": str(prd_path),
                "workspace": str(workspace),
                **invocation.metadata(),
            },
        )

        if cli_result.get("dispatched"):
            return ToolResult.ok(
                confirmation=f"Dispatched batch (job_id={cli_result.get('job_id', '?')})",
                data=cli_result,
            )
        return ToolResult.failed(
            cli_result.get("error") or "talon batch dispatch failed",
            data=cli_result,
        )

    @tool(
        name="talon_status",
        description=(
            "Check status of Talon jobs (running, completed, failed). "
            "Merges locally-dispatched jobs (the durable registry) with jobs "
            "only observed via the shared observability store — e.g. runs "
            "driven by a Claude Code session or a peer host. Each job carries a "
            "'source' provenance field ('registry' vs 'observability'). Pass "
            "source='observability' to list only externally-driven jobs, or "
            "source='registry' for only local ones."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon status",
    )
    async def talon_status(self, source: Optional[str] = None) -> ToolResult:
        """Check status of Talon jobs across the registry and observability.

        Reaps any background CLI subprocess that has finished since the last
        call (updates ``status`` to ``complete`` or ``failed`` based on
        returncode), reconciles A2A jobs, then merges the durable registry with
        the observability-event-backed view of externally-dispatched jobs.

        Args:
            source: Provenance filter. ``None``/``"all"`` merges both sources;
                ``"registry"``/``"local"`` restricts to locally-dispatched jobs;
                ``"observability"``/``"external"`` restricts to jobs seen only
                via observability events. Every returned job carries its own
                ``source`` marker regardless of the filter.
        """
        # Validate the provenance filter up front so a typo fails fast (with the
        # valid values) instead of silently reporting zero jobs — and before any
        # reap/reconcile side effects run.
        norm = (source or "").strip().lower()
        if (
            norm not in _SOURCE_FILTER_ALL
            and norm not in _SOURCE_FILTER_REGISTRY
            and norm not in _SOURCE_FILTER_OBSERVABILITY
        ):
            valid = sorted(
                v
                for v in (
                    _SOURCE_FILTER_ALL
                    | _SOURCE_FILTER_REGISTRY
                    | _SOURCE_FILTER_OBSERVABILITY
                )
                if v
            )
            return ToolResult.failed(
                f"Unknown source filter {source!r}. Valid values: "
                f"{', '.join(valid)} (or omit for all).",
                data={"source_filter": norm, "valid_sources": valid},
            )

        # Pull in any CLI jobs persisted before a restart so they are
        # visible again even without an in-memory process handle.
        self._reload_persisted_jobs()

        # Reap finished background CLI jobs
        jobs_changed = False
        for jid, info in list(self._jobs.items()):
            if self._reap_cli_job(info):
                jobs_changed = True

        if jobs_changed:
            self._persist_jobs()

        # Reconcile A2A-dispatched jobs against Talon's task_store.
        # Mesh used to do this via inbox polling for complete/reject
        # messages; the A2A equivalent is querying the recipient's
        # task by id and mapping its TaskState back to coordinator
        # status. Without this, talon_status reports method=a2a jobs
        # as "dispatched" forever and downstream pollers
        # (FeatureFeature implementation gate, workflows runner) hang
        # waiting for a completion that never surfaces. Codex P2 on
        # PR #1368.
        host_url = self._discover_host_url()
        a2a_jobs_to_check = [
            (jid, info) for jid, info in self._jobs.items()
            if info.get("method") == "a2a"
            and info.get("status") in _RUNNING_TALON_STATES
        ]
        if host_url and a2a_jobs_to_check:
            a2a_changed = False
            for jid, info in a2a_jobs_to_check:
                if await self._reconcile_a2a_job(jid, info, host_url):
                    a2a_changed = True
            if a2a_changed:
                self._persist_jobs()

        def _public(info: Dict[str, Any]) -> Dict[str, Any]:
            # Strip non-serialisable fields (the asyncio Process handle).
            return {
                k: v for k, v in info.items()
                if k != "process"
            }

        # Local registry jobs, tagged with their provenance. These are the
        # ground truth for anything this coordinator dispatched.
        registry_jobs = [
            {**_public(info), "id": jid, "source": JOB_SOURCE_REGISTRY}
            for jid, info in self._jobs.items()
        ]
        # Correlation keys the registry already owns: raw job ids AND the
        # ``repo#issue`` a fleet ``workflow_run_id`` uses. A fleet event stream
        # keys on ``workflow_run_id`` (``owner/repo#issue``), which never equals
        # a registry job id, so dedup has to compare the derived ``repo#issue``
        # too — otherwise a job this coordinator dispatched would double-count as
        # both a registry job and an observability job.
        registry_corr = {job["id"] for job in registry_jobs}
        for info in self._jobs.values():
            repo = info.get("repo")
            issue = info.get("issue")
            if repo and issue is not None:
                registry_corr.add(f"{repo}#{issue}")

        include_registry = (
            norm in _SOURCE_FILTER_ALL or norm in _SOURCE_FILTER_REGISTRY
        )
        include_observability = (
            norm in _SOURCE_FILTER_ALL or norm in _SOURCE_FILTER_OBSERVABILITY
        )

        # Observability-observed jobs. A job the registry already knows about
        # wins (its live process handle / exit sidecar is authoritative), so
        # only observability-ONLY jobs are merged in — matched by job key and by
        # ``repo#issue`` correlation — no double-counting, no regression to
        # registry-based reporting. A registry-only request skips the fleet
        # query entirely.
        observability_jobs: List[Dict[str, Any]] = []
        if include_observability:
            observed = await self._observability_talon_jobs()
            observability_jobs = [
                desc for jid, desc in observed.items()
                if jid not in registry_corr
                and desc.get("correlation") not in registry_corr
            ]

        selected: List[Dict[str, Any]] = []
        if include_registry:
            selected.extend(registry_jobs)
        if include_observability:
            selected.extend(observability_jobs)

        running: List[Dict[str, Any]] = []
        done: List[Dict[str, Any]] = []
        # Anything that classified as neither (defensive; observability phases
        # all map into the two buckets) is still surfaced rather than dropped.
        other: List[Dict[str, Any]] = []
        for job in selected:
            status = job.get("status")
            if status in _RUNNING_TALON_STATES:
                running.append(job)
            elif status in _TERMINAL_TALON_STATES:
                done.append(job)
            else:
                other.append(job)

        data = {
            "running": len(running),
            "completed": len(done),
            "jobs": running + done + other,
            "registry_count": sum(
                1 for job in selected if job.get("source") == JOB_SOURCE_REGISTRY
            ),
            "observability_count": sum(
                1 for job in selected
                if job.get("source") == JOB_SOURCE_OBSERVABILITY
            ),
            "source_filter": norm or "all",
        }
        return ToolResult.ok(
            confirmation=(
                f"Talon jobs: running={len(running)}, completed={len(done)} "
                f"(registry={data['registry_count']}, "
                f"observability={data['observability_count']})"
            ),
            data=data,
        )

    def _self_agent_name(self) -> Optional[str]:
        """This coordinator's owning agent identity, or ``None`` when unbound.

        ``talon_job`` lifecycle events are recorded on the shared
        ``a2a_observability`` table tagged (in the ``agent_name`` column) with
        the agent that OWNS the job — the dispatcher whose ``talon_status`` is
        meant to surface it — not the worker that ran it. In a shared multi-agent
        deployment (one PostgreSQL pool, one table) every agent's store reads the
        SAME rows, so ``talon_status`` MUST scope its read to this agent's own
        identity; otherwise one agent's status would leak other tenants' external
        job ids, repos, issues, PRs, and workflow metadata (the cross-agent
        disclosure this hardening closes).

        Resolving to ``None`` — an unbound / single-tenant standalone agent —
        leaves the query unscoped, because there is no other tenant to leak to.
        This mirrors the graph store's unbound ``1 = 1`` ownership scope, and
        reads ``_agent_name`` first / ``agent_name`` second exactly like the rest
        of the coordinator (see ``_observability_context``).
        """
        agent = getattr(self, "agent", None)
        if agent is not None:
            for attr in ("_agent_name", "agent_name"):
                candidate = getattr(agent, attr, None)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return None

    async def _observability_talon_jobs(
        self,
        *,
        since_minutes: int = _OBSERVABILITY_LOOKBACK_MINUTES,
        limit: int = _OBSERVABILITY_EVENT_LIMIT,
    ) -> Dict[str, Dict[str, Any]]:
        """Reduce ``talon_job`` observability events into per-job status (#2646).

        Reads the agent's observability store (``ObservabilityStore.query_events``
        over the ``a2a_observability`` table) for ``talon_job`` lifecycle events
        and folds them into a ``job_id -> descriptor`` map. This is the
        observability-backed status source of #2646: it surfaces Talon jobs
        dispatched OUTSIDE this coordinator's local registry (Claude Code
        sessions, peer hosts) that would otherwise be invisible to
        ``talon_status``.

        The read is **tenant-scoped**: it filters ``query_events`` by this
        agent's own ``agent_name`` (see :meth:`_self_agent_name`), so on a shared
        store one agent never sees another agent's Talon jobs. An unbound
        standalone agent (no name) reads unscoped — there is no peer tenant to
        isolate from.

        Read-only and best-effort: when no observability store is attached, or the
        query fails for any reason, it degrades to an empty map rather than
        breaking status reporting — a stock deployment whose store holds no
        ``talon_job`` rows (owned by this agent) simply reports registry jobs
        only.
        """
        store = getattr(self.agent, "observability_store", None)
        if store is None:
            return {}
        since = datetime.now(timezone.utc) - timedelta(
            minutes=max(1, int(since_minutes))
        )
        try:
            events = await store.query_events(
                agent_name=self._self_agent_name(),
                event_type=_TALON_JOB_EVENT_TYPE,
                since=since,
                limit=int(limit),
            )
        except Exception as exc:  # best-effort: never break status reporting
            logger.debug("talon_job observability query failed: %s", exc)
            return {}
        return self._reduce_observability_talon_jobs(events)

    def _reduce_observability_talon_jobs(
        self, events: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Fold ``talon_job`` observability events into a ``job_id -> descriptor``.

        Pure over the ``ObservabilityEvent`` shape (attribute access:
        ``.metadata``/``.timestamp``/``.agent_name``/``.session_id``). A Talon run
        is correlated by ``metadata["talon_job_id"]``; events lacking one are
        ignored. The lifecycle phase lives in ``metadata["talon_event"]``
        (claimed/started/iteration/completed/failed) and defines status.

        The store returns newest-first, so the first event seen per job defines the
        current status while older events backfill any correlation
        (repo/issue/orchestrator/stage/pr/workflow_run_id) the latest event lacked,
        extend the first-seen timestamp, and bump the observed-event count.
        """
        jobs: Dict[str, Dict[str, Any]] = {}
        for event in events or []:
            meta = getattr(event, "metadata", None)
            meta = meta if isinstance(meta, dict) else {}

            job_id = meta.get("talon_job_id")
            if not isinstance(job_id, str) or not job_id.strip():
                # No job id: nothing to correlate a job on.
                continue
            job_id = job_id.strip()

            phase = meta.get("talon_event")
            ts = getattr(event, "timestamp", None)
            ts_iso = str(ts) if ts is not None else None

            repo = meta.get("repo") or None
            issue = _coerce_issue(meta.get("issue"))
            correlation = (
                f"{repo}#{issue}" if repo and issue is not None else None
            )
            orchestrator = meta.get("orchestrator")
            stage = meta.get("stage")
            pr = meta.get("pr")
            workflow_run_id = meta.get("workflow_run_id")

            existing = jobs.get(job_id)
            if existing is None:
                # First (== latest, newest-first) event defines status.
                jobs[job_id] = {
                    "id": job_id,
                    "source": JOB_SOURCE_OBSERVABILITY,
                    "status": _talon_event_status(phase),
                    "talon_event": phase,
                    "repo": repo,
                    "issue": issue,
                    "pr": pr,
                    "orchestrator": orchestrator,
                    "stage": stage,
                    "workflow_run_id": workflow_run_id,
                    "correlation": correlation,
                    "agent_name": getattr(event, "agent_name", None),
                    "session_id": getattr(event, "session_id", None),
                    "last_event_at": ts_iso,
                    "first_event_at": ts_iso,
                    "observed_events": 1,
                }
                continue

            # Older event for a known job: bump the count, keep the earliest
            # timestamp, and backfill any correlation the latest event lacked.
            existing["observed_events"] += 1
            if ts_iso is not None:
                existing["first_event_at"] = ts_iso
            for key, value in (
                ("repo", repo),
                ("issue", issue),
                ("pr", pr),
                ("orchestrator", orchestrator),
                ("stage", stage),
                ("workflow_run_id", workflow_run_id),
                ("correlation", correlation),
            ):
                if existing.get(key) is None and value is not None:
                    existing[key] = value

        return jobs

    @tool(
        name="talon_job_log",
        description=(
            "Tail the log file of a dispatched Talon job. Use the "
            "job_id returned by talon_claim. Read-only."
        ),
        category=ToolCategory.UTILITY,
        command_prefix="!talon job-log",
    )
    async def talon_job_log(
        self, job_id: str, lines: int = 200,
    ) -> ToolResult:
        """Return the last ``lines`` lines of a job's combined log."""
        # Reload persisted CLI jobs so logs are tail-able after restart.
        self._reload_persisted_jobs()
        info = self._jobs.get(job_id)
        if not info:
            return ToolResult.failed(
                f"Unknown job_id: {job_id}",
                data={"job_id": job_id},
            )

        log_path = info.get("log_path")
        if not log_path or not os.path.isfile(log_path):
            return ToolResult.failed(
                f"Log file missing: {log_path}",
                data={"job_id": job_id, "log_path": log_path},
            )

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-max(1, int(lines)):]
        except OSError as e:
            return ToolResult.failed(str(e), data={"job_id": job_id})

        return ToolResult.ok(
            confirmation=(
                f"Tailed {len(tail)} line(s) from {log_path} "
                f"(status={info.get('status')}, rc={info.get('returncode')})"
            ),
            data={
                "success": True,
                "job_id": job_id,
                "log_path": log_path,
                "status": info.get("status"),
                "returncode": info.get("returncode"),
                "lines": len(tail),
                "content": "".join(tail),
            },
        )

    @tool(
        name="talon_pause",
        description="Pause the autonomous Talon loop (kill switch).",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon pause",
    )
    async def talon_pause(self) -> ToolResult:
        """Pause the Talon dispatch loop.

        Disables the signal_dispatch scheduler so no new work is
        dispatched until resumed with !talon resume.
        """
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'remove_schedule'):
            scheduler.remove_schedule("signal_dispatch")
            return ToolResult.ok(
                confirmation="Talon dispatch paused. Use !talon resume to restart.",
                data={"paused": True, "message": "Talon dispatch paused. Use !talon resume to restart."},
            )
        return ToolResult.failed(
            "Scheduler not available",
            data={"paused": False, "error": "Scheduler not available"},
        )

    @tool(
        name="talon_resume",
        description="Resume the autonomous Talon loop after pause.",
        category=ToolCategory.SYSTEM,
        command_prefix="!talon resume",
    )
    async def talon_resume(self) -> ToolResult:
        """Resume the Talon dispatch loop after a pause."""
        scheduler = getattr(self.agent, '_scheduler', None)
        if scheduler and hasattr(scheduler, 'add_schedule'):
            scheduler.add_schedule("signal_dispatch", "5 8 * * *", "signal_dispatch")
            return ToolResult.ok(
                confirmation="Talon dispatch resumed at 08:05 daily.",
                data={"resumed": True, "message": "Talon dispatch resumed at 08:05 daily."},
            )
        return ToolResult.failed(
            "Scheduler not available",
            data={"resumed": False, "error": "Scheduler not available"},
        )

    @tool(
        name="talon_health",
        description=(
            "Check whether kestrel-talon is reachable and runnable: "
            "binary discoverable, subprocess env clean, executes "
            "``--help`` successfully. Read-only, no dispatch."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!talon health",
    )
    async def talon_health(self) -> ToolResult:
        """Smoke-test the talon CLI path without dispatching real work.

        Returns a structured report covering:

        * ``binary``: where the executable lives (or why we couldn't
          find it).
        * ``env``: which Anthropic keys were stripped, and which
          GitHub token name was found. The actual token VALUE is
          never returned.
        * ``execute``: result of ``kestrel-talon --help``: returncode,
          first line of stdout, and stderr summary if it failed.

        Use this BEFORE ``talon_claim`` on a real issue — most past
        dispatch failures came from missing/wrong env, and this
        check catches them in seconds without burning a Claude Max
        session or touching GitHub.
        """
        report: Dict[str, Any] = {"healthy": False}

        def _wrap(r: Dict[str, Any]) -> ToolResult:
            """Final wrap: healthy → OK; any failure → ERROR with the
            most-specific reason in the error string so the LLM can't
            narrate "talon healthy" off a binary-not-found body."""
            if r.get("healthy"):
                return ToolResult.ok(
                    confirmation=(
                        f"kestrel-talon healthy "
                        f"(binary={r.get('binary', {}).get('path')})"
                    ),
                    data=r,
                )
            # Pull the most specific error available. Stage dicts may
            # carry ``error`` directly OR a non-zero ``returncode`` +
            # ``stderr_tail`` (the --help-fails-mid-startup path); both
            # need to surface, otherwise the agent loses the actual
            # cause that talon_health exists to diagnose.
            for stage in ("execute", "env", "binary"):
                stage_data = r.get(stage)
                if not isinstance(stage_data, dict):
                    continue
                if stage_data.get("error"):
                    return ToolResult.failed(
                        f"talon health {stage}: {stage_data['error']}",
                        data=r,
                    )
                if stage == "execute" and stage_data.get("ok") is False:
                    rc = stage_data.get("returncode")
                    stderr_tail = (stage_data.get("stderr_tail") or "").strip()
                    detail = stderr_tail[-300:] if stderr_tail else "no stderr"
                    return ToolResult.failed(
                        f"talon health execute: --help exited rc={rc}; "
                        f"stderr_tail={detail!r}",
                        data=r,
                    )
            return ToolResult.failed(
                "talon health check failed",
                data=r,
            )

        # 1. Binary discovery
        talon_bin = self._find_talon_bin()
        if not talon_bin:
            report["binary"] = {
                "found": False,
                "error": (
                    "kestrel-talon not found. Set KESTREL_TALON_BIN, "
                    "`uv sync` it into this venv, or place a sibling "
                    "checkout at ../kestrel-talon with its own .venv."
                ),
            }
            return _wrap(report)
        report["binary"] = {"found": True, "path": talon_bin}

        # 2. Env cleanliness
        stripped = [
            k for k in self._ANTHROPIC_KEYS_TO_STRIP if k in os.environ
        ]
        env_report: Dict[str, Any] = {
            "stripped_anthropic_keys": stripped,
        }
        try:
            built_env = self._build_subprocess_env()
        except RuntimeError as e:
            env_report["error"] = str(e)
            report["env"] = env_report
            return _wrap(report)

        token_source = next(
            (
                name
                for name in ("GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PAT")
                if os.environ.get(name)
            ),
            None,
        )
        env_report["github_token_source"] = token_source
        env_report["env_var_count"] = len(built_env)
        report["env"] = env_report

        # 3. Execute --help to prove the binary is runnable AND that
        # talon's own startup (which imports Claude Agent SDK) doesn't
        # crash with our env. Bounded: --help returns in < 5s.
        cmd = [talon_bin, "--help"]
        try:
            result = await run_bounded_subprocess(
                cmd,
                env=built_env,
                timeout=15,
                max_output_bytes=_TALON_CAPTURE_LIMIT_BYTES,
            )
        except Exception as e:
            report["execute"] = {"ok": False, "error": redact_secrets(str(e))}
            return _wrap(report)

        if result.timed_out:
            report["execute"] = {
                "ok": False,
                "error": "kestrel-talon --help timed out after 15s",
            }
            return _wrap(report)

        out = redact_secrets(result.stdout.decode(errors="replace"))
        err = redact_secrets(result.stderr.decode(errors="replace"))
        first_out_line = next((ln for ln in out.splitlines() if ln.strip()), "")
        report["execute"] = {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "first_line": first_out_line[:200],
            "stderr_tail": err[-400:] if err else "",
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        }

        report["healthy"] = bool(report["execute"]["ok"])
        return _wrap(report)

    # ------------------------------------------------------------------
    # Internal: orchestrator identity + workflow correlation (talon#53)
    # ------------------------------------------------------------------

    def _observability_context(self) -> Dict[str, str]:
        """The three frozen observability keys for the current dispatch.

        Built at the coordinator boundary so EVERY outgoing talon
        invocation — direct tool calls and workflow-stage signal-source
        dispatches alike — carries the same correlation fields:

        * ``KESTREL_OBSERVABILITY_ORCHESTRATOR``: the friendly agent name
          driving this dispatch. Read off the owning agent; when the
          dispatch runs inside a workflow stage the workflow executes on
          this same agent, so its identity is the workflow's
          owning/triggering identity too. Omitted (never empty-string)
          when no real agent name is available — downstream treats null
          as "Direct".
        * ``KESTREL_OBSERVABILITY_WORKFLOW_RUN_ID`` /
          ``KESTREL_OBSERVABILITY_STAGE``: only present when the dispatch
          is running inside a ``kind == "workflow.stage"`` signal dispatch
          (the workflows runner sets ``Signal.session_id = run.run_id``).
          Read from the dispatcher's per-task current-signal context so
          the coordinator needs no cooperation from individual handlers.
        """
        ctx: Dict[str, str] = {}

        agent = getattr(self, "agent", None)
        # Real KestrelAgent instances store the friendly name on
        # ``_agent_name`` (there is no ``agent_name`` property) — the same
        # attribute the rest of the codebase reads first (SecurityFeature,
        # approval_queue, lifecycle_checks, codex adapters). ``agent_name``
        # is kept as a fallback for stubs/alternate agent shapes. Mock/stub
        # agents can expose a non-string attribute; only a real, non-empty
        # string is a usable identity — otherwise the key is omitted (never
        # empty-string) and downstream renders "Direct".
        if agent is not None:
            for attr in ("_agent_name", "agent_name"):
                candidate = getattr(agent, attr, None)
                if isinstance(candidate, str) and candidate.strip():
                    ctx[OBSERVABILITY_ORCHESTRATOR_KEY] = candidate.strip()
                    break

        try:
            from kestrel_sovereign.signals.context import get_current_signal
            signal = get_current_signal()
        except Exception:  # pragma: no cover - defensive import guard
            signal = None
        if signal is not None and getattr(signal, "kind", None) == "workflow.stage":
            run_id = getattr(signal, "session_id", None)
            if isinstance(run_id, str) and run_id:
                ctx[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] = run_id
            stage = self._stage_name_from_signal(signal)
            if stage:
                ctx[OBSERVABILITY_STAGE_KEY] = stage
        return ctx

    @staticmethod
    def _stage_name_from_signal(signal: Any) -> Optional[str]:
        """Best-effort workflow stage name off a ``workflow.stage`` Signal.

        The workflows runner injects ``workflow_stage_name`` into the payload
        only for ``feature_features.*`` sources, so fall back to the causation
        frame it always emits (``source = "workflow.<spec>.<stage>"``): strip
        the known ``workflow.<spec>.`` prefix by splitting on the first two
        dots so a stage name that itself contains dots (the workflows name
        grammar permits them, e.g. ``deploy.v2``) survives intact.
        """
        payload = getattr(signal, "payload", None)
        if isinstance(payload, dict):
            stage = payload.get("workflow_stage_name")
            if isinstance(stage, str) and stage:
                return stage
        for frame in reversed(list(getattr(signal, "causation_chain", None) or [])):
            frame_source = getattr(frame, "source", None)
            if isinstance(frame_source, str) and frame_source.startswith(
                "workflow."
            ):
                parts = frame_source.split(".", 2)
                if len(parts) == 3 and parts[2]:
                    return parts[2]
        return None

    # ------------------------------------------------------------------
    # Internal: Mesh dispatch (preferred)
    # ------------------------------------------------------------------

    async def _dispatch_via_a2a(
        self, repo: str, issue_number: int, title: str = ""
    ) -> Dict[str, Any]:
        """Submit an A2A task to Talon — the replacement for mesh dispatch.

        Builds a ``TaskSendParams``-shaped payload and POSTs it to
        Talon's ``/api/agent/tasks/send`` endpoint. The receiving side
        creates the task, fires ``a2a.task_submitted`` so Talon's
        cognition loop wakes up, and acts on the assignment. Same
        functional contract as the prior mesh dispatch but with
        durable persistence (TaskStore vs in-memory inbox), lifecycle
        states (SUBMITTED → WORKING → COMPLETED), and signal-driven
        wakeup (#1366 / #1367)."""
        from uuid import uuid4

        host_url = self._discover_host_url()
        if not host_url:
            return {"dispatched": False, "reason": "no_a2a_host"}

        sender = getattr(self.agent, "agent_name", None) or getattr(
            self.agent, "did", "kestrel",
        )
        task_id = uuid4().hex
        session_id = uuid4().hex
        body = (
            f"Assignment: implement issue {repo}#{issue_number} "
            f"({title or 'no title'}). Take the work and report back "
            f"via task completion."
        )
        metadata: Dict[str, Any] = {
            "sender": sender,
            "skill": "workflow.assign",
            "repo": repo,
            "issue_number": issue_number,
            "issue_title": title or f"#{issue_number}",
        }
        # Orchestrator identity + workflow correlation (kestrel-talon#53):
        # carried as structured metadata fields on the A2A message so the
        # talon daemon can map them into its invocation context.
        metadata.update(self._observability_context())
        payload = json.dumps({
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": body}],
            },
            "metadata": metadata,
        }).encode("utf-8")

        url = f"{host_url}/api/agents/talon/api/agent/tasks/send"
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=10).read()
            )
            self._jobs[task_id] = {
                "repo": repo, "issue": issue_number,
                "status": "dispatched", "method": "a2a",
            }
            return {
                "dispatched": True, "method": "a2a",
                "task_id": task_id, "repo": repo, "issue": issue_number,
            }
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.debug(f"A2A dispatch failed: {e}")
            return {"dispatched": False, "reason": "a2a_unavailable", "error": str(e)}

    # ------------------------------------------------------------------
    # Internal: CLI fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _find_talon_bin() -> Optional[str]:
        """Locate the kestrel-talon executable.

        Search order:
        1. ``KESTREL_TALON_BIN`` env var (explicit override).
        2. ``shutil.which`` against the running process's PATH (works
           when ``uv sync`` installed kestrel-talon into this venv).
        3. Sibling-checkout convention used in dev: a ``kestrel-talon``
           directory next to ``kestrel-sovereign`` with its own
           ``.venv/bin/kestrel-talon``. Required because Jason's
           workflow keeps talon as a separate ``uv run`` source tree
           rather than installed into the kestrel venv.
        """
        override = os.environ.get("KESTREL_TALON_BIN")
        if override and os.path.isfile(override) and os.access(override, os.X_OK):
            return override

        on_path = shutil.which("kestrel-talon")
        if on_path:
            return on_path

        # Sibling layout: parents = talon, features, kestrel_sovereign,
        # kestrel-sovereign (project root), then projects/. The sibling
        # checkout lives next to the project root.
        sibling = (
            Path(__file__).resolve().parents[4]
            / "kestrel-talon"
            / ".venv"
            / "bin"
            / "kestrel-talon"
        )
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)

        return None

    # Anthropic credentials kestrel-talon must NOT see — its Claude
    # Agent SDK call chain auto-uses Claude Max OAuth from ``~/.claude``,
    # but only if no API-key env var is set. If kestrel-sovereign was
    # launched with ``ANTHROPIC_API_KEY`` (which it usually is — the
    # main agent uses it for its own LLM calls), passing that env
    # straight through silently flips talon onto API-key billing
    # AND breaks any "I am running as user X" identity assertions.
    #
    # See ``feedback_kestrel_talon.md``: "API key is specifically
    # stripped." Order matters too — Claude Agent SDK merges parent
    # ``os.environ`` after we hand it our env dict, so the talon
    # binary itself further mutates ``os.environ`` at runtime; we
    # only need to make sure our subprocess starts clean.
    _ANTHROPIC_KEYS_TO_STRIP = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    )

    @staticmethod
    def _build_subprocess_env(
        backend: str = "claude",
        auth_lane: str = "oauth",
    ) -> Dict[str, str]:
        """Construct the env dict for a kestrel-talon subprocess.

        Backend-specific sanitization keeps Talon from inheriting credentials
        unrelated to the selected runtime, and verifies a GitHub token is
        present. Raises ``RuntimeError`` with an actionable message if a
        required var is missing; callers convert that into a structured
        ``dispatched=False`` response.
        """
        try:
            env, _stripped = sanitize_env_for_backend(
                normalize_backend(backend) or "claude",
                normalize_auth_lane(auth_lane) or "oauth",
            )
            return env
        except TalonRuntimeError as e:
            raise RuntimeError(str(e)) from e

    def _job_log_dir(self) -> Path:
        """Where job log files live.

        Per-agent under the agent's data directory when available
        (``<storage_path>/talon_jobs/``) so logs survive process
        restarts and aren't shared across agents in the multi_agent.
        Falls back to ``/tmp`` when the agent has no storage_path
        (test stubs, ephemeral runs).
        """
        storage_path = getattr(self.agent, "storage_path", None) if self.agent else None
        if storage_path:
            base = Path(storage_path).parent / "talon_jobs"
        else:
            base = Path("/tmp/kestrel_talon_jobs")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _jobs_registry_path(self) -> Path:
        """Durable registry of CLI-background job metadata.

        Lives next to the per-agent log files so it survives Kestrel
        restarts (``<storage_path>/talon_jobs/jobs.json``).
        """
        return self._job_log_dir() / "jobs.json"

    def _job_exit_path(self, job_id: str) -> Path:
        """Sidecar file the dispatch wrapper writes its exit code to.

        Used after a Kestrel restart to recover the true exit status
        of CLI background jobs that finished while no in-process handle
        was awaiting them.
        """
        return self._job_log_dir() / f"{job_id}.exit"

    @staticmethod
    def _tail_job_log(path: Any, lines: int = 20) -> str:
        """Best-effort tail of a job's combined log file.

        Used by :class:`TalonWaitable` to attach a short context snippet to
        the COGNITION signal the reconciler emits, so the agent does not have to
        call ``talon_job_log`` as a follow-up tool just to see what
        happened. Returns ``""`` on any read error — the signal is
        still useful without it.
        """
        if not path:
            return ""
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                buf = f.readlines()
        except (OSError, UnicodeDecodeError):
            return ""
        if not buf:
            return ""
        return "".join(buf[-lines:])

    def _reap_cli_job(self, info: Dict[str, Any]) -> bool:
        """Refresh a single ``cli_background`` job's status in place.

        Returns ``True`` when the job's persisted state changed so the
        caller knows to re-persist the registry. No-op (returns
        ``False``) for jobs that are not ``cli_background`` or are no
        longer in ``dispatched``/``running``. Shared by ``talon_status``
        and the ``TalonWaitable`` provider so the reaping logic lives in
        one place.
        """
        if info.get("method") != "cli_background":
            return False
        if info.get("status") not in ("dispatched", "running"):
            return False
        proc = info.get("process")
        if proc is None:
            # Reloaded after restart: no live handle. The sidecar exit
            # file is the authoritative source of truth — if it exists,
            # the wrapper recorded the exit code. If not, fall back to
            # pid liveness (best-effort; PID reuse can produce a false
            # 'running').
            rc = self._read_exit_sidecar(info.get("exit_path"))
            if rc is not None:
                info["status"] = "complete" if rc == 0 else "failed"
                info["returncode"] = rc
                info.setdefault(
                    "completed_at",
                    datetime.now(timezone.utc).isoformat(),
                )
                return True
            if self._pid_alive(info.get("pid")):
                info["status"] = "running"
                return False
            # Process gone, no sidecar: status genuinely unknown. Do NOT
            # claim 'complete' — that would lie about failures that
            # exited before the wrapper wrote.
            info["status"] = "finished_unknown"
            info.setdefault("returncode", None)
            info.setdefault(
                "completed_at",
                datetime.now(timezone.utc).isoformat(),
            )
            return True
        rc = proc.returncode
        if rc is None:
            info["status"] = "running"
            return False
        info["status"] = "complete" if rc == 0 else "failed"
        info["returncode"] = rc
        info["completed_at"] = datetime.now(timezone.utc).isoformat()
        return True

    async def _reconcile_a2a_job(
        self, jid: str, info: Dict[str, Any], host_url: str
    ) -> bool:
        """Refresh a single ``a2a`` job's status against Talon's task_store.

        Queries the recipient's A2A task by id and maps its TaskState
        back to coordinator status: SUBMITTED/WORKING→running,
        COMPLETED→complete, FAILED/CANCELED→failed. Returns ``True`` when
        the persisted state changed so the caller knows to re-persist the
        registry. No-op (returns ``False``) for jobs that are not ``a2a``,
        are no longer ``dispatched``/``running``, or when the network
        query fails. Shared by ``talon_status`` and the ``TalonWaitable``
        provider so the A2A reconciliation lives in one place — mirroring ``_reap_cli_job``
        for the CLI transport.
        """
        if info.get("method") != "a2a":
            return False
        if info.get("status") not in ("dispatched", "running"):
            return False
        if not host_url:
            return False
        url = f"{host_url}/api/agents/talon/api/agent/tasks/{jid}"
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            raw = await asyncio.to_thread(
                lambda: urllib.request.urlopen(req, timeout=5).read()
            )
            task_payload = json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
            # Network blip / Talon offline / task not yet visible to
            # Talon's task_store. Leave the coordinator's row in its
            # current state; the next poll will retry.
            return False
        state = (
            task_payload.get("status", {}) or {}
        ).get("state") or task_payload.get("status")
        # Map A2A TaskState → coordinator status. SUBMITTED and WORKING
        # stay "running"; COMPLETED→complete; FAILED/CANCELED→failed.
        if state in ("submitted", "working"):
            if info.get("status") != "running":
                info["status"] = "running"
                return True
            return False
        if state == "completed":
            info["status"] = "complete"
            info["completed_at"] = datetime.now(timezone.utc).isoformat()
            return True
        if state in ("failed", "canceled"):
            info["status"] = "failed"
            info["completed_at"] = datetime.now(timezone.utc).isoformat()
            err_msg = (
                (task_payload.get("status", {}) or {})
                .get("message", {}) or {}
            )
            info["error"] = err_msg.get("text") or state
            return True
        return False

    @staticmethod
    def _read_exit_sidecar(path: Any) -> Optional[int]:
        """Read an exit-code sidecar; ``None`` if absent or malformed."""
        if not path:
            return None
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                content = f.read().strip()
        except (OSError, UnicodeDecodeError):
            return None
        if not content:
            return None
        try:
            return int(content)
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        """Return True if ``pid`` names a live process (signal 0 probe)."""
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return False
        if pid_int <= 0:
            return False
        try:
            os.kill(pid_int, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but is owned by another user — still alive.
            return True
        except OSError:
            return False
        return True

    def _persist_jobs(self) -> bool:
        """Write CLI-background job metadata to the durable registry.

        Excludes non-serialisable fields (the asyncio ``process``
        handle) so ``talon_status`` and ``talon_job_log`` keep working
        after a feature/server restart. Returns whether the atomic replace
        succeeded; initial dispatch uses this as its ownership-transfer gate.
        """
        registry: Dict[str, Any] = {}
        for jid, info in self._jobs.items():
            if info.get("method") != "cli_background":
                continue
            registry[jid] = {
                k: v for k, v in info.items() if k != "process"
            }
        # Use a unique tmp file per writer so concurrent _persist_jobs
        # calls from sibling feature instances cannot clobber each
        # other's pre-replace temp content. os.replace() is atomic on
        # POSIX so the final rename remains race-free.
        tmp_path: Optional[str] = None
        try:
            path = self._jobs_registry_path()
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(registry, f)
            os.replace(tmp_path, path)
            tmp_path = None
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to persist talon job registry: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return False

    def _reload_persisted_jobs(self) -> None:
        """Merge durably-persisted CLI jobs into the in-memory map.

        In-process jobs (which still hold a live ``process`` handle)
        win; only job_ids absent from memory are reloaded so a
        restarted feature regains status/log visibility.
        """
        try:
            path = self._jobs_registry_path()
        except (TypeError, OSError) as e:
            # Test stubs / discovery harness can construct the
            # feature with a Mock agent whose ``storage_path`` is
            # not a valid path. Skip reload silently in that case;
            # real agents always have a real storage_path.
            logger.debug("Talon job registry path unavailable: %s", e)
            return
        if not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read talon job registry: {e}")
            return
        if not isinstance(registry, dict):
            return
        for jid, info in registry.items():
            if jid in self._jobs or not isinstance(info, dict):
                continue
            # Reloaded jobs have no live process handle.
            info.pop("process", None)
            self._jobs[jid] = info
            # Reconstruct the workflow run→job binding (#2303, fifth pass) so a
            # restart between the talon_run and verify_ci stages still resolves
            # this run's own job. In-process bindings win (setdefault).
            run_id = info.get("workflow_run_id")
            if isinstance(run_id, str) and run_id:
                self._pipeline_run_jobs.setdefault(run_id, jid)

    async def _survey_stalled_talon_jobs(
        self, stale_days: Any = 3,
    ) -> List[Dict[str, Any]]:
        """Discover live stalled Talon jobs for ``fleet_stalled_sweep`` (#2200).

        A job is *stalled* when it is still ``dispatched``/``running`` and its
        ``started_at`` is older than ``stale_days`` (a missing/unparseable
        timestamp is treated as stalled — surfacing it for review is safe).
        Read-only observation: it returns descriptor dicts and never dispatches,
        closes, or mutates anything. The irreversible ``a2a_repair_dispatch``
        stage will still refuse to act on these without explicit per-run repair
        targets and fresh approval.
        """
        try:
            self._reload_persisted_jobs()
        except Exception as exc:  # noqa: BLE001 - survey degrades, never aborts
            logger.debug("stalled-work survey could not reload jobs: %s", exc)
        try:
            days = max(0.0, float(stale_days))
        except (TypeError, ValueError):
            days = 3.0
        threshold = datetime.now(timezone.utc) - timedelta(days=days)

        stalled: List[Dict[str, Any]] = []
        for jid, info in list(self._jobs.items()):
            if not isinstance(info, dict):
                continue
            if info.get("status") not in ("dispatched", "running"):
                continue
            started_raw = info.get("started_at")
            started = None
            if isinstance(started_raw, str):
                try:
                    started = datetime.fromisoformat(started_raw)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                except ValueError:
                    started = None
            if started is not None and started > threshold:
                continue  # recent activity — not stalled yet
            stalled.append({
                "id": jid,
                "kind": "talon_job",
                "label": info.get("label"),
                "repo": info.get("repo"),
                "issue": info.get("issue"),
                "status": info.get("status"),
                "started_at": started_raw,
            })
        return stalled

    async def _dispatch_via_cli_background(
        self,
        args: List[str],
        label: str,
        env: Optional[Dict[str, str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Launch kestrel-talon as a background subprocess and return.

        The subprocess writes combined stdout+stderr to a log file
        the caller can tail later via ``talon_job_log(job_id)``.
        Job metadata is tracked in ``self._jobs`` so ``talon_status``
        can reap finished processes and report them.
        """
        talon_bin = self._find_talon_bin()
        if not talon_bin:
            return {
                "dispatched": False,
                "error": (
                    "kestrel-talon not found. Set KESTREL_TALON_BIN, install "
                    "kestrel-talon into the kestrel-sovereign venv "
                    "(`uv sync`), or place a sibling checkout at "
                    "../kestrel-talon with its own .venv."
                ),
            }

        if env is None:
            try:
                env = self._build_subprocess_env()
            except RuntimeError as e:
                return {"dispatched": False, "error": str(e)}

        # Orchestrator identity + workflow correlation (kestrel-talon#53):
        # set as process env vars on the spawned talon process. Stamped here
        # — the single CLI dispatch funnel — so claim, batch, iterate, and
        # every workflow-stage source carry them without per-caller wiring.
        observability = self._observability_context()
        if observability:
            env = {**env, **observability}

        job_id = uuid.uuid4().hex
        log_path = self._job_log_dir() / f"{job_id}.log"
        exit_path = self._job_exit_path(job_id)
        cmd = [talon_bin] + args

        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            return {"dispatched": False, "error": f"Cannot open log file: {e}"}

        # Wrap with sh -c so the exit code is written to a sidecar file
        # atomically when the subprocess terminates. The sidecar is the
        # authoritative source of truth for status after a Kestrel
        # restart, when no live process handle exists.
        exit_quoted = shlex.quote(str(exit_path))
        wrapper_script = (
            f'"$@"; rc=$?; printf "%s" "$rc" > {exit_quoted}.tmp && '
            f'mv {exit_quoted}.tmp {exit_quoted}; exit "$rc"'
        )
        wrapped = ["sh", "-c", wrapper_script, "_talon_wrapper"] + cmd

        try:
            proc = await start_async_process(
                wrapped,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=Path(
                    os.environ.get("KESTREL_TALON_CWD")
                    or str(_DEFAULT_PROJECT_PARENT)
                ),
            )
        except Exception as e:
            return {"dispatched": False, "error": redact_secrets(str(e))}
        finally:
            # Close our handle on success, failure, or caller cancellation —
            # the child gets its own descriptor. Otherwise the file stays open
            # in the host for the subprocess lifetime and launch cancellation
            # leaks the parent's descriptor.
            log_file.close()

        info: Dict[str, Any] = {
            "method": "cli_background",
            "label": label,
            "command": redact_secrets(" ".join(cmd)),
            "status": "dispatched",
            "pid": proc.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "exit_path": str(exit_path),
            "process": proc,
        }
        if extra_meta:
            info.update(extra_meta)
        self._jobs[job_id] = info
        if not self._persist_jobs():
            # The process becomes intentionally independent of this feature
            # only after its durable row exists. Before that ownership
            # transfer, a launch must fail closed and tear down the complete
            # private group; otherwise a Kestrel restart would orphan an
            # invisible Talon job.
            self._jobs.pop(job_id, None)
            try:
                await terminate_process_tree(proc)
            except Exception as cleanup_error:  # noqa: BLE001
                logger.error(
                    "Failed to clean up unpersisted Talon job %s: %s",
                    job_id,
                    cleanup_error,
                    exc_info=True,
                )
                return {
                    "dispatched": False,
                    "error": (
                        "Could not persist Talon job ownership, and process-tree "
                        f"cleanup failed: {redact_secrets(str(cleanup_error))}"
                    ),
                }
            for artifact in (log_path, exit_path, Path(f"{exit_path}.tmp")):
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Could not remove unowned Talon artifact %s",
                        artifact,
                        exc_info=True,
                    )
            return {
                "dispatched": False,
                "error": "Could not persist durable Talon job ownership",
            }

        logger.info(
            f"Dispatched talon job {job_id} (pid={proc.pid}, "
            f"label={label}, log={log_path})"
        )

        return {
            "dispatched": True,
            "method": "cli_background",
            "job_id": job_id,
            "pid": proc.pid,
            "log_path": str(log_path),
            "label": label,
            "started_at": info["started_at"],
            "next_step": (
                "Poll talon_status to see when it completes, or "
                "talon_job_log(job_id) to see live output."
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_host_url(self) -> Optional[str]:
        """Discover the multi_agent host URL."""
        host_url = os.environ.get("KESTREL_HOST_URL")
        if host_url:
            return host_url.rstrip("/")

        from kestrel_sovereign.paths import project_dir
        for candidate in [
            Path.cwd() / "multi_agent.toml",
            project_dir() / "multi_agent.toml",
        ]:
            if candidate.exists():
                try:
                    import toml
                    data = toml.load(candidate)
                    port = data.get("host", {}).get("port", 8888)
                    return f"http://localhost:{port}"
                except Exception as e:
                    logger.debug(f"Could not read {candidate}: {e}")
        return None
