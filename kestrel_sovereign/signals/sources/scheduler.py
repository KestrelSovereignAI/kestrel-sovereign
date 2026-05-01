"""Source registrations for built-in cron tasks (Phase 4 of #889).

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

`signal_dispatch` is named confusingly but is unrelated to this PR —
it's an existing scheduler tool that dispatches work to Talon. ACTION.
"""

from __future__ import annotations

import json
import logging
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
# a new built-in scheduled task: append here, ensure the underlying tool
# exists, and (if ACTION) wire the handler in SchedulerFeature.
CRON_TASKS: list[tuple[str, SignalMode, frozenset[ResourceLock]]] = [
    # Pure ops — no LLM, no shared state beyond what the handler touches.
    ("backup_snapshot", SignalMode.ACTION, frozenset()),
    # External dispatch — sends work to Talon; agent storage untouched.
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
    # Reflection workflow returning a structured Dict. Read-only of
    # conversation; doesn't write back.
    ("reflect", SignalMode.ARTIFACT, frozenset()),
    # Consolidation writes episodes/patterns into memory storage.
    # Owner confirmed ARTIFACT (#893): no follow-up cognition triggered.
    ("memory_consolidate", SignalMode.ARTIFACT, frozenset({ResourceLock.MEMORY})),
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


def _make_action_handler(lookup: ToolLookup, task_name: str) -> ActionHandler:
    """Return an ACTION handler that runs `lookup(task_name, payload)`."""

    async def handler(payload: dict) -> Any:
        return await lookup(task_name, payload)

    return handler


def _make_artifact_handler(
    lookup: ToolLookup, task_name: str
) -> ArtifactHandler:
    """Return an ARTIFACT handler that runs `lookup(task_name, signal.payload)`
    and returns the tool's output as the artifact."""

    async def handler(signal: Signal) -> Any:
        return await lookup(task_name, signal.payload)

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
            tasks not in `builtin_handlers`. The existing scheduler's
            tool-search logic is the production implementation; tests
            inject a fake.
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
            handler: ActionHandler | None = builtin_handlers.get(
                task_name, _make_action_handler(tool_lookup, task_name)
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
