"""Source registrations for core cron-capable tasks (Phase 4 of #889).

Each cron task gets its own SourceRegistration. The mode classification
captures the per-task semantic — see SIGNAL_DISPATCHER.md §"The three
modes" for the framing:

    ACTION    — deterministic side effect (no LLM turn, no follow-up
                cognition). Examples: backup_snapshot, trash_retention,
                training_cycle.
    ARTIFACT  — produces an artifact (text or dict) via a feature
                workflow that may make one or more LLM calls internally.
                Does NOT enter conversation history; does NOT trigger
                follow-up cognition. Examples: morning_signal, reflect,
                memory_consolidate.

The handler factory (`_make_tool_handler`) is shared across all
registrations whose underlying implementation is "look up a feature tool
by name and execute it" — that's the entire dispatch pattern of the
existing scheduler. Built-in tasks (`backup_snapshot`, `trash_retention`)
that don't go through tool lookup get bespoke handlers wired by the
SchedulerFeature.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Awaitable, Callable

from kestrel_sdk.signals import (
    ActionHandler,
    ArtifactHandler,
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
)
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome

logger = logging.getLogger(__name__)


# Source name prefix — `cron.<task_name>` keeps cron sources legible in
# signal_log queries and lets `kestrel_sovereign/signals/sources/` grep
# turn up every cron-driven entry point at once.
SOURCE_PREFIX = "cron."


# ---------------------------------------------------------------------------
# Per-task classification
# ---------------------------------------------------------------------------


# (task_name, mode, resources_touched). Source registrations are built
# from this table at agent init by `build_cron_registrations()`. To add
# a new signalized scheduled task: append here, ensure the underlying tool
# exists, and (if it has no feature tool) wire a built-in ACTION handler in
# SchedulerFeature. Membership does not imply that core auto-seeds a row.
CRON_TASKS: list[tuple[str, SignalMode, frozenset[ResourceLock]]] = [
    # Pure ops — no LLM, no shared state beyond what the handler touches.
    ("backup_snapshot", SignalMode.ACTION, frozenset()),
    # Provider-neutral strategic dispatch. It remains user-schedulable even
    # though core no longer auto-seeds it; the ACTION source preserves the
    # normal cron signal log/span and routes to the live feature tool.
    ("signal_dispatch", SignalMode.ACTION, frozenset()),
    # Touches storage to purge soft-deleted rows past retention. MEMORY
    # is the closest existing lock; refining the lock taxonomy is a
    # follow-up.
    ("trash_retention", SignalMode.ACTION, frozenset({ResourceLock.MEMORY})),
    # Long-running LLM ops that modify model/training state. ACTION
    # because it doesn't return cognition for the bird to act on.
    ("training_cycle", SignalMode.ACTION, frozenset({ResourceLock.MEMORY})),
    # Feature workflow returning briefing text. Read-only.
    ("morning_signal", SignalMode.ARTIFACT, frozenset()),
    # Reflection workflow returning a structured Dict. ReflectionFeature
    # persists each session (`_persist_reflection` writes session +
    # insights rows), so it shares storage state with memory_consolidate
    # and any other reflection persistence path. Coarse MEMORY lock
    # keeps them serialized; a finer reflection-specific lock is a
    # follow-up if the over-serialization with memory_consolidate ever
    # bites in practice.
    ("reflect", SignalMode.ARTIFACT, frozenset({ResourceLock.MEMORY})),
    # Consolidation writes episodes/patterns into memory storage.
    # Owner confirmed ARTIFACT (#893): no follow-up cognition triggered.
    # Still a schedulable tool, but no longer auto-seeded — the nightly `sleep`
    # cycle (below) now owns memory maintenance (#1674 P3).
    ("memory_consolidate", SignalMode.ARTIFACT, frozenset({ResourceLock.MEMORY})),
    # Nightly `sleep` cycle (#1674 P3) — the single memory-maintenance cron:
    # reflection (via sleep_hooks), consolidation, and the forgetting
    # deletion tier through MemorySystem.consolidate(). Holds MEMORY (writes
    # episodes + deletes decayed ones). ACTION (NOT artifact): sleep has a
    # built-in handler (SchedulerFeature._handle_sleep) and no feature tool, and
    # build_cron_registrations only wires builtin_handlers for ACTION tasks — an
    # ARTIFACT registration would fall through to tool lookup and fail as
    # "Unknown task: sleep". Supersedes the auto-seeded memory_consolidate +
    # reflect crons.
    ("sleep", SignalMode.ACTION, frozenset({ResourceLock.MEMORY})),
    # Generic wait reconciler (Wave 2 of #1860). ACTION — no LLM cost.
    # Enumerates every MonitorableWaitable provider in agent.wait_registry,
    # polls each provider's in-flight handles, and ENQUEUES one COGNITION
    # signal per terminal-state transition. The COGNITION wake comes from
    # that downstream signal, not from the reconcile task itself. This is
    # the generic monitor for all providers; it's core (no feature gate).
    # Built-in handler: run_wait_reconcile(agent).
    ("wait_reconcile", SignalMode.ACTION, frozenset()),
    # Restart coordinator (#1512). ACTION — no LLM cost. Scans the
    # restart_requests table, evaluates safety, and spawns a detached
    # subprocess to execute the restart. Idle by default; only fires
    # when a row is pending.
    ("restart_coordinator", SignalMode.ACTION, frozenset()),
    # GitHub PR/issue watcher (#1618). ACTION — no LLM cost on the poll
    # itself. Fetches the PR's current state, fingerprints the watched
    # fields, and ENQUEUES one github.pr_activity COGNITION signal only
    # when a relevant change is detected (state/comment/check/merge).
    # The COGNITION wake comes from that downstream signal, not from the
    # watch task itself. Built-in handler:
    # SchedulerFeature._run_github_pr_watch. Per-watch config (repo, pr,
    # triggers, notify) travels in the scheduled task's args_json.
    ("github_pr_watch", SignalMode.ACTION, frozenset()),
    # Ecosystem discovery watcher (#2281). ACTION — no LLM cost in the
    # scheduler handler itself beyond the delegated discovery tool. It
    # fingerprints stale-work / red-CI findings and ENQUEUES one
    # ecosystem.discovery_findings COGNITION signal only when actionable
    # findings are new, changed, or just resolved. The COGNITION wake
    # comes from that downstream signal, not from the watch task itself.
    ("ecosystem_discovery_watch", SignalMode.ACTION, frozenset()),
    # Bootstrap watchdog (#378). ACTION — no LLM cost. Checks whether a
    # never-contacted agent is still bootstrap_state=pending past the timeout
    # and escalates status to stale_bootstrap.
    ("bootstrap_timeout_check", SignalMode.ACTION, frozenset({ResourceLock.MEMORY})),
]


def cron_source_name(task_name: str) -> str:
    return f"{SOURCE_PREFIX}{task_name}"


# ---------------------------------------------------------------------------
# Schema / redaction policy
# ---------------------------------------------------------------------------


def _cron_schema(payload: dict) -> dict:
    """Cron task args are arbitrary tool kwargs — accept any dict.
    Per-task schema validation is the tool's responsibility (it raises
    on bad kwargs, which the dispatcher captures as FAILED).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"cron payload must be a dict, got {type(payload).__name__}")
    return payload


def _cron_redact(payload: dict) -> str:
    """Cron payloads are operator-controlled config strings (cron
    expression args from scheduled_tasks.args_json). They're safe to
    summarize as the JSON itself, but cap length for log hygiene.
    """
    try:
        text = json.dumps(payload, default=str, sort_keys=True)
    except Exception:
        text = repr(payload)
    if len(text) > 200:
        text = text[:200] + "...(truncated)"
    return f"args={text}"


_REDACTION = RedactionPolicy(
    summarize=_cron_redact,
    store_raw_trusted=False,
    redact_caller_identifier=True,
)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


# Tool lookup signature shared by all "find a feature tool by name and
# call it" cron sources. Builtin tasks (backup_snapshot, trash_retention)
# get their own handlers wired by SchedulerFeature.
ToolLookup = Callable[[str, dict], Awaitable[Any]]


def _require_successful_task_result(
    task_name: str,
    result: Any,
    *,
    decode_json_envelope: bool = False,
) -> Any:
    """Turn a structured task failure into a failed scheduler dispatch.

    Permission blocks remain normal handler results: the scheduler runner needs
    their structured outcome to pause the schedule and persist operator-facing
    recovery guidance without making an expected policy decision look like a
    dispatcher error. ``decode_json_envelope`` is reserved for bespoke built-in
    handlers, whose documented wire result is a JSON object string. Feature
    tools may legitimately return arbitrary string artifacts, so their strings
    must never be interpreted as scheduler control envelopes.
    """
    if isinstance(result, ScheduledTaskOutcome):
        if result.status == "blocked":
            return result
        # ``result_text`` is the task_execution_log artifact and may contain a
        # complete sleep report, third-party hook payloads, or governed
        # semantic-maintenance maps. Never splice it into an exception: the
        # dispatcher persists exception text in ``signal_log.error`` and emits
        # it to ERROR logs without the result-summary redaction/cap boundary.
        raise RuntimeError(
            f"scheduled task {task_name} returned {result.status}"
        )

    evaluated_result = result
    if decode_json_envelope and isinstance(result, str):
        try:
            decoded_result = json.loads(result)
        except json.JSONDecodeError:
            decoded_result = None
        if isinstance(decoded_result, Mapping):
            evaluated_result = decoded_result

    failed = (
        isinstance(evaluated_result, ToolResult)
        and evaluated_result.status is ToolResultStatus.ERROR
    )
    if isinstance(evaluated_result, Mapping):
        # DynamicTool serializes ToolResult before scheduler lookup returns;
        # its legacy exception wrapper uses ``success=False``.  Both shapes are
        # terminal failures, not successful cron artifacts.
        status = evaluated_result.get("status")
        non_terminal = (
            status == "blocked"
            or evaluated_result.get("blocked") not in (None, False, "")
            or evaluated_result.get("skipped") not in (None, False, "")
        )
        # Watchers use ``blocked`` plus ``error`` for expected retryable states,
        # and built-ins use ``skipped`` for inapplicable configuration/policy
        # states.  A bare built-in ``error`` remains terminal, but an explicit
        # ``success=True`` verdict owns partial-operation diagnostics such as a
        # successful consolidation followed by an optional export failure.
        failed = (
            not non_terminal
            and (
                status in (ToolResultStatus.ERROR, ToolResultStatus.ERROR.value)
                or evaluated_result.get("success") is False
                or (
                    decode_json_envelope
                    and evaluated_result.get("error") not in (None, False, "")
                    and evaluated_result.get("success") is not True
                )
            )
        )
    if failed:
        # ToolResult.error and built-in JSON ``error`` values are untrusted
        # result bodies: they may contain provider/memory detail or a QueryError
        # with SQL. The dispatcher persists exception text outside the bounded
        # result-summary channel, so expose only the task identity and verdict.
        raise RuntimeError(f"scheduled tool {task_name} failed")
    return result


def _prepare_scheduled_tool_result(task_name: str, result: Any) -> Any:
    """Validate and serialize one feature-tool result at the source boundary."""
    checked = _require_successful_task_result(task_name, result)
    if isinstance(checked, ScheduledTaskOutcome) or isinstance(checked, str):
        return checked

    # Keep the established task_execution_log JSON contract without forcing
    # the feature lookup itself to serialize before this boundary can inspect
    # ToolResult/mapping failures.
    from kestrel_sovereign.features.base import _serialize_tool_result

    return json.dumps(_serialize_tool_result(checked), default=str)


def _wrap_builtin_action_handler(
    handler: ActionHandler,
    task_name: str,
) -> ActionHandler:
    """Apply the scheduled-result contract to a bespoke ACTION handler."""

    async def checked_handler(payload: dict) -> Any:
        try:
            result = await handler(payload)
        except Exception:
            # Keep the actionable traceback in the trusted local log while the
            # dispatcher/audit boundary receives fixed, content-free text.
            logger.exception("Scheduled built-in task %s raised", task_name)
            raise RuntimeError(
                f"scheduled task {task_name} raised"
            ) from None
        return _require_successful_task_result(
            task_name, result, decode_json_envelope=True
        )

    return checked_handler


def _make_action_handler(lookup: ToolLookup, task_name: str) -> ActionHandler:
    """Return an ACTION handler that runs `lookup(task_name, payload)`."""

    async def handler(payload: dict) -> Any:
        try:
            result = await lookup(task_name, payload)
        except Exception:
            logger.exception("Scheduled tool %s raised", task_name)
            raise RuntimeError(
                f"scheduled tool {task_name} raised"
            ) from None
        return _prepare_scheduled_tool_result(task_name, result)

    return handler


def _make_artifact_handler(
    lookup: ToolLookup, task_name: str
) -> ArtifactHandler:
    """Return an ARTIFACT handler that runs `lookup(task_name, signal.payload)`
    and returns the tool's output as the artifact."""

    async def handler(signal: Signal) -> Any:
        try:
            result = await lookup(task_name, signal.payload)
        except Exception:
            logger.exception("Scheduled tool %s raised", task_name)
            raise RuntimeError(
                f"scheduled tool {task_name} raised"
            ) from None
        return _prepare_scheduled_tool_result(task_name, result)

    return handler


# ---------------------------------------------------------------------------
# Registration builder
# ---------------------------------------------------------------------------


def build_cron_registrations(
    *,
    tool_lookup: ToolLookup,
    builtin_handlers: dict[str, ActionHandler] | None = None,
) -> list[SourceRegistration]:
    """Build SourceRegistration instances for all built-in cron tasks.

    Args:
        tool_lookup: Async fn `(task_name, args) -> result`. Used by all
            tasks not in `builtin_handlers`. It returns the raw feature-tool
            result; the source handler owns structured failure validation and
            legacy JSON serialization for every caller of this builder.
        builtin_handlers: Per-task ACTION handlers that bypass tool
            lookup entirely. Used for `backup_snapshot` (calls
            sync.force_snapshot directly) and `trash_retention` (calls
            SchedulerFeature._run_trash_retention). Keys are bare task
            names (without the cron. prefix).

    Returns:
        One SourceRegistration per entry in CRON_TASKS.
    """
    builtin_handlers = builtin_handlers or {}
    registrations: list[SourceRegistration] = []

    for task_name, mode, resources in CRON_TASKS:
        # Pick the handler. Builtins win over tool lookup.
        if mode == SignalMode.ACTION:
            builtin_handler = builtin_handlers.get(task_name)
            handler: ActionHandler | None = (
                _make_action_handler(tool_lookup, task_name)
                if builtin_handler is None
                else _wrap_builtin_action_handler(builtin_handler, task_name)
            )
            artifact_handler = None
        else:  # ARTIFACT
            handler = None
            artifact_handler = _make_artifact_handler(tool_lookup, task_name)

        registrations.append(
            SourceRegistration(
                name=cron_source_name(task_name),
                schema=_cron_schema,
                default_mode=mode,
                allowed_modes=frozenset({mode}),
                handler=handler,
                artifact_handler=artifact_handler,
                # COGNITION sources need a prompt template; cron sources
                # never use COGNITION (no allowed mode includes it).
                prompt_template=None,
                trust=Trust.TRUSTED,
                # Cron expressions ARE the rate limit. Disable dispatcher
                # throttling to avoid two-layer drops where a bursty cron
                # would mysteriously skip ticks.
                rate_limit=RateLimit(),
                # No coalescing — each cron firing is a distinct intent.
                coalescing_window=None,
                # Cron tasks run regardless of "quiet hours" (they're
                # operational). attention_policy with no quiet_hours
                # means always-allowed.
                attention_policy=AttentionPolicy(),
                resources=resources,
                allow_self_loops=False,
                log_redaction=_REDACTION,
                # 30 days — long enough to debug a regressing weekly
                # cron, short enough to keep signal_log lean.
                retention_days=30,
            )
        )

    return registrations
