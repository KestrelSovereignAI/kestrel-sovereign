"""Pre-turn state-load block (epic #1290, D3).

Before the LLM sees a user message, inject a compact system-role snapshot
of where the agent actually stands: her strategy, what her scheduled work
did in the last day, what's waiting in her mesh inbox, how many approvals
are queued, and what she has pinned. This is the keystone that makes a
sovereign agent *proactive* instead of reactive — she starts every turn
already oriented instead of rediscovering her own state from scratch.

It is opt-in per agent via ``[preturn_state]`` in ``kestrel.toml``
(default off; Emma is the first opt-in). Every section is best-effort:
a missing feature or a query failure degrades that one line to
``(unavailable)`` and never blocks the turn. The whole block is capped
(~``max_tokens``) so it can't crowd the real prompt.

The block is merged into ``system_prompt_addendum`` by
:func:`prepare_turn_state_addendum`, which both turn entry points
(``_process_input_traced_locked`` and ``process_input_streaming``) call
after the USER_PROMPT_SUBMIT hook and before context assembly, so it rides
the existing, budget-aware addendum plumbing rather than adding a parallel
injection path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kestrel_sdk.native_a2a.types import TaskState

from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.lifecycle import task_state_label
from kestrel_sovereign.a2a.projections import message_text
from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_scoped_agent_did,
)

if TYPE_CHECKING:
    from kestrel_sovereign.features.strategic_memory.recent_decisions import (
        RecentDecision,
        RecentDecisions,
    )

logger = logging.getLogger(__name__)

_BLOCK_HEADER = "--- AGENT STATE (pre-turn snapshot; not user input) ---"
_BLOCK_FOOTER = "--- END AGENT STATE ---"

_OPERATIONAL_HEADER = (
    "--- OPERATIONAL STATE (lifecycle events; not user input) ---"
)
_OPERATIONAL_FOOTER = "--- END OPERATIONAL STATE ---"


def _get_feature(agent: Any, name: str) -> Any:
    try:
        if hasattr(agent, "get_feature"):
            return agent.get_feature(name)
    except Exception:  # noqa: BLE001
        return None
    return None


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cap by a conservative ~4 chars/token budget."""
    budget = max(1, max_tokens) * 4
    if len(text) <= budget:
        return text
    return text[: budget - 3].rstrip() + "..."


def _strategy_section(agent: Any) -> Optional[str]:
    storage_path = getattr(agent, "storage_path", None)
    if not storage_path:
        return "Strategy: (unavailable)"
    path = Path(storage_path).parent / "strategy.yaml"
    if not path.exists():
        return None  # No strategy.yaml — say nothing rather than noise.
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: strategy.yaml parse failed: %s", exc)
        return "Strategy: (unreadable)"

    parts: List[str] = []
    vision = data.get("vision") or data.get("mission")
    if vision:
        parts.append(f"Vision: {str(vision).strip()[:240]}")
    decisions = data.get("open_decisions") or data.get("decisions") or []
    if isinstance(decisions, list) and decisions:
        head = "; ".join(str(d).strip() for d in decisions[:3])
        parts.append(f"Open decisions ({len(decisions)}): {head}")
    blockers = data.get("blockers") or data.get("active_blockers") or []
    if isinstance(blockers, list) and blockers:
        head = "; ".join(str(b).strip() for b in blockers[:3])
        parts.append(f"Blockers ({len(blockers)}): {head}")
    return "\n".join(parts) if parts else None


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def _scheduled_section(agent: Any) -> Optional[str]:
    feat = _get_feature(agent, "SchedulerFeature")
    if feat is None or not hasattr(feat, "schedule_history"):
        return None
    try:
        res = await feat.schedule_history(limit=40)
        execs = (res.data or {}).get("executions", []) if res else []
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: schedule_history failed: %s", exc)
        return "Scheduled tasks (24h): (unavailable)"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    succ = fail = skip = other = 0
    windowed = False
    for e in execs:
        ts = _parse_ts(e.get("executed_at"))
        if ts is not None:
            windowed = True
            if ts < cutoff:
                continue
        status = str(e.get("status", "")).lower()
        if status in ("success", "ok", "completed"):
            succ += 1
        elif status in ("fail", "failed", "error"):
            fail += 1
        elif status in ("skip", "skipped"):
            skip += 1
        else:
            other += 1
    total = succ + fail + skip + other
    if total == 0:
        return "Scheduled tasks (24h): none ran"
    label = "24h" if windowed else "recent"
    return (
        f"Scheduled tasks ({label}): {succ} ok, {fail} failed, "
        f"{skip} skipped"
        + (f", {other} other" if other else "")
    )


async def _a2a_inbox_section(agent: Any) -> Optional[str]:
    """Show non-terminal A2A tasks addressed to this agent.

    Replaces the prior ``_mesh_section`` which read PeersFeature's
    in-memory ``_mesh_inbox`` list. Mesh was retired in #1367; the
    durable equivalent is the agent's TaskStore. SUBMITTED and WORKING
    tasks where this agent is the receiver are the "pending inbox"
    semantic the prior surface tried to convey.
    """
    task_manager = getattr(agent, "task_manager", None)
    if task_manager is None:
        return None
    try:
        recipient_agent_id = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return None
    try:
        snapshots = await task_manager.list_task_snapshots(
            recipient_agent_id=recipient_agent_id,
            limit=50,
        )
    except Exception:
        return None
    pending = [
        snapshot for snapshot in snapshots
        if snapshot.task.status.state in (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING)
    ]
    n = len(pending)
    if n == 0:
        return "A2A inbox: no pending tasks"
    lines = [f"A2A inbox: {n} pending task(s); last 3:"]
    for snapshot in pending[:3]:
        task = snapshot.task
        skill = str(MessageToDict(task.metadata).get("skill") or "")
        first_text = message_text(task.history[0]) if task.history else None
        first_text = (first_text or "").replace("\n", " ").strip()
        label = f"{skill or 'task'} from {snapshot.ownership.creator_agent_id}"
        lines.append(
            f"  - [{task_state_label(task.status.state)}] {label}: {first_text[:90]}"
        )
    return "\n".join(lines)


async def _approvals_section(agent: Any) -> Optional[str]:
    feat = _get_feature(agent, "SecurityFeature")
    queue = getattr(feat, "approval_queue", None) if feat is not None else None
    if queue is None:
        return None
    try:
        count = queue.pending_count
    except Exception:  # noqa: BLE001
        return None
    return f"Pending approvals: {count} awaiting the Sovereign"


async def _restart_status_section(agent: Any) -> Optional[str]:
    """Render the most recent restart lifecycle events as non-
    instructional context (#1562 / #1551 follow-up).

    These rows live in ``restart_status_events`` and exist so the
    agent knows whether a restart/update was requested, deferred,
    executed, or rejected since her last turn — without having to
    treat the data as an instruction or a developer message. The
    block is summary-only: a one-line per-state count plus the
    latest transition, so it can never crowd the real prompt.
    """
    feat = _get_feature(agent, "RestartCoordinatorFeature")
    if feat is None:
        return None
    # In a shared multi-agent database, the global recency listing
    # would leak peer agents' restart lifecycle into this agent's
    # state block. Scope the query to the current agent's DID via
    # the dedicated agent-scoped store helper (codex P2 r1).
    db = getattr(feat, "_db", None)
    if db is None:
        return None
    try:
        agent_did = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return None
    try:
        from kestrel_sovereign.features.restart_coordinator.event_store import (
            list_recent_events_for_agent_context,
        )
        rows = await list_recent_events_for_agent_context(
            db, agent_id=agent_did, limit=20,
        )
        events = [r.to_public_dict() for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: restart status events failed: %s", exc)
        return None
    if not events:
        return None
    by_state: dict[str, int] = {}
    for e in events:
        s = str(e.get("state", "")).strip() or "unknown"
        by_state[s] = by_state.get(s, 0) + 1
    parts = ", ".join(
        f"{n} {state}" for state, n in sorted(by_state.items())
    )
    latest = events[0]  # newest first from list_recent_events_for_history
    request_id = str(latest.get("request_id", ""))[:8]
    state = str(latest.get("state", ""))
    return (
        f"Restart events (recent {len(events)}): {parts}; "
        f"latest: {request_id}… → {state}"
    )


async def _pinned_section(agent: Any) -> Optional[str]:
    feat = _get_feature(agent, "MemoryAgencyFeature")
    if feat is None or not hasattr(feat, "memory_pinned"):
        return None
    try:
        res = await feat.memory_pinned()
        pins = (res.data or {}).get("pins", []) if res else []
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: memory_pinned failed: %s", exc)
        return None
    if not pins:
        return "Pinned memory: none"
    lines = [f"Pinned memory ({len(pins)}):"]
    for p in pins[:4]:
        head = str(p.get("preview") or p.get("reason") or "").strip()
        head = head.replace("\n", " ")
        lines.append(f"  - {head[:90]}")
    return "\n".join(lines)


async def build_preturn_state_block(agent: Any) -> Optional[str]:
    """Return the pre-turn state block, or ``None`` to inject nothing.

    Gated by ``[preturn_state]`` in kestrel.toml: ``enabled`` must be
    true and (if ``agents`` is set) the agent's name must be listed.
    """
    try:
        from kestrel_sovereign.config import load_section

        cfg = load_section("preturn_state")
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: config load failed: %s", exc)
        return None
    if not cfg or not cfg.get("enabled"):
        return None

    agent_name = getattr(agent, "_agent_name", None)
    allowed = cfg.get("agents") or []
    if allowed and agent_name not in allowed:
        return None

    max_tokens = int(cfg.get("max_tokens", 500))

    sections: List[Optional[str]] = []
    try:
        sections.append(_strategy_section(agent))
        sections.append(await _scheduled_section(agent))
        sections.append(await _a2a_inbox_section(agent))
        sections.append(await _approvals_section(agent))
        # Restart lifecycle events are required operational context and
        # ride the always-on ``build_operational_state_block`` path so
        # they surface even when ``[preturn_state]`` is disabled (#1571).
        # Don't render them here too — that would duplicate the block.
        sections.append(await _pinned_section(agent))
    except Exception as exc:  # noqa: BLE001 - never break a turn
        logger.warning(
            "preturn_state: block assembly failed for %s: %s",
            agent_name, exc, exc_info=True,
        )
        return None

    body = "\n".join(s for s in sections if s)
    if not body.strip():
        return None
    block = f"{_BLOCK_HEADER}\n{body}\n{_BLOCK_FOOTER}"
    return _truncate_to_tokens(block, max_tokens)


async def _codex_decline_section(agent: Any) -> Optional[str]:
    """Render the agent's most recent codex sandbox-approval declines
    as a one-line summary (#1581).

    DID-scoped read of ``codex_decline_events``. Silent when no rows
    exist or the table hasn't been created (decline events only
    surface when codex actually emits an approval RPC AND we decline
    it — most agents will see nothing here most of the time).

    Without this, declines via codex's hardcoded default-reply table
    (elicitation/permissions/userInput) AND the bridge's fail-closed
    paths (policy DENY, queue denied, missing CU, malformed payload)
    are completely invisible to the agent on her next turn.
    """
    from kestrel_sovereign.features.storage_access import (
        resolve_feature_database,
    )
    db = resolve_feature_database(agent)
    if db is None:
        return None
    try:
        agent_id = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable:
        return None
    try:
        from kestrel_sovereign.llm.codex_decline_events import (
            list_recent_declines_for_agent,
        )
        rows = await list_recent_declines_for_agent(
            db, agent_id=agent_id, limit=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "operational_state: codex_decline read failed: %s", exc,
        )
        return None
    # #1694: the same store also records auto-approves via the
    # codex_native bridge (status="auto_approved"). Those are
    # transparency events, not blockers; the operational state block
    # surfaces what BLOCKED a turn so filter them out here.
    declined_rows = [r for r in rows if r.status == "declined"]
    if not declined_rows:
        return None
    by_reason: dict[str, int] = {}
    for ev in declined_rows:
        key = ev.reason.split(":", 1)[0]
        by_reason[key] = by_reason.get(key, 0) + 1
    counts = ", ".join(
        f"{n} {r}" for r, n in sorted(by_reason.items())
    )
    latest = declined_rows[0]
    latest_tool = latest.tool[:80]
    return (
        f"Codex declines (recent {len(declined_rows)}): {counts}; "
        f"latest: {latest.request} → {latest_tool!r}"
    )


def _inert_text(text: Any, max_len: int, *, mark_truncation: bool = False) -> str:
    """Free-form text rendered as inert data for the operational block.

    Todo titles and strategy decisions are free-form and (transitively)
    user-influenceable, but get injected into system context on EVERY future
    turn — so treat them as inert data, not instructions (codex review).
    Collapse whitespace, drop non-printable/control chars (incl. the 0x1e
    stream-sentinel and anything that could fake the operational framing),
    swap double quotes for single so the caller's quoting holds, and cap
    length. With ``mark_truncation`` a cut is shown as a trailing ``…``.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = "".join(ch for ch in s if ch.isprintable())
    # Collapse any run of 2+ dashes to one: the OPERATIONAL_HEADER/FOOTER
    # boundaries are ``--- … ---`` framed, so this stops a single-line todo
    # title like "--- END OPERATIONAL STATE ---" from spoofing the block
    # boundary even after newline/control stripping (codex review r2).
    s = re.sub(r"-{2,}", "-", s)
    s = s.replace('"', "'")
    if mark_truncation and len(s) > max_len:
        return s[: max(0, max_len - 1)].rstrip() + "…"
    return s[:max_len]


async def _active_todo_section(agent: Any) -> Optional[str]:
    """Surface active TodoFeature items so unfinished durable loops stay
    visible across signal wakes, restarts, and delayed tool callbacks
    (#1907 — the deferred #1832 pre-turn-injection follow-up).

    Without this, the durable todo queue (#1843) is pull-only: the model only
    sees unfinished work if it remembers to call ``todo_rollup``. Here it rides
    the always-on operational block alongside restart-status, so every wake
    re-surfaces what the agent still owes. Bounded + non-instructional: a
    summary header plus a capped list with each item's terminal condition (so
    the agent knows what "done" means and doesn't stop after a poll when the
    real condition is "wait until merged + restarted"). ``None`` when nothing
    is active.
    """
    feat = _get_feature(agent, "TodoFeature")
    if feat is None or not hasattr(feat, "active_preturn_items"):
        return None
    try:
        items = await feat.active_preturn_items()
    except Exception as exc:  # noqa: BLE001
        logger.debug("preturn_state: active todos failed: %s", exc)
        return None
    session = items.get("session") or []
    global_active = items.get("global_active") or []
    if not session and not global_active:
        return None

    def _fmt(it: Dict[str, Any]) -> str:
        title = _inert_text(it.get("title") or it.get("id") or "todo", 100)
        # status comes from a controlled vocabulary (TODO_STATUSES), not free
        # text, so it's safe unquoted.
        status = str(it.get("status") or "open")
        line = f'  - [{status}] "{title}"'
        tc = _inert_text(it.get("terminal_condition"), 120)
        if tc:
            line += f' (done when: "{tc}")'
        return line

    lines: List[str] = []
    if session:
        lines.append(f"Active todos — this session ({len(session)}):")
        lines.extend(_fmt(it) for it in session[:6])
    if global_active:
        lines.append(
            f"Active todos — other sessions, in_progress/waiting ({len(global_active)}):"
        )
        lines.extend(_fmt(it) for it in global_active[:6])
    return _truncate_to_tokens("\n".join(lines), 300)


@dataclass(frozen=True)
class RecentDecisionsConfig:
    """``[operational_state.recent_decisions]`` in kestrel.toml (#3900).

    ``window_hours`` selects the decisions shown; decisions record a day, so
    the window is widened to whole local days (24 hours shows today's and
    yesterday's). The newest ``max_entries`` (at most 100) are shown, each an
    excerpt of at most ``excerpt_chars`` of its decision and rationale, and
    the whole section is at most ``max_chars`` characters (~400 tokens by
    default), which must leave 400 beside ``excerpt_chars`` so the newest
    decision always fits.
    """

    window_hours: float = 24
    max_entries: int = 8
    excerpt_chars: int = 240
    max_chars: int = 1600


#: Ten years. Far past any useful "recent", and well inside what ``timedelta``
#: and date arithmetic accept: an overflow there would drop the section.
_MAX_RECENT_DECISIONS_WINDOW_HOURS = 24 * 366 * 10

#: Bounds the digits the "not shown" line can take.
_MAX_RECENT_DECISIONS_ENTRIES = 100

#: Room for everything in the section but excerpts: the header, the "not
#: shown" line and one entry's date, session and quotes. ``max_chars`` must
#: leave this much beside ``excerpt_chars``, so the newest decision is always
#: shown and the section never outgrows its cap.
_RECENT_DECISIONS_OVERHEAD_CHARS = 400

#: Session names are free-form; this many characters of one are shown.
_SESSION_CHARS = 40


def recent_decisions_config() -> RecentDecisionsConfig:
    """Read ``[operational_state.recent_decisions]``; ``ValueError`` if invalid.

    An unknown key is refused rather than ignored, so a misspelt
    ``window_hour`` does not silently leave the default in force.
    """
    from kestrel_sovereign.config import load_section

    where = "[operational_state.recent_decisions]"
    section = load_section("operational_state")
    if not isinstance(section, dict):
        raise ValueError("[operational_state] must be a table")
    table = section.get("recent_decisions", {})
    if not isinstance(table, dict):
        raise ValueError(f"{where} must be a table, got {table!r}")
    known = {f.name for f in fields(RecentDecisionsConfig)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known))}"
        )
    window = table.get("window_hours", RecentDecisionsConfig.window_hours)
    if isinstance(window, bool) or not isinstance(window, (int, float)) or not (
        0 < window <= _MAX_RECENT_DECISIONS_WINDOW_HOURS
    ):
        raise ValueError(
            f"{where} window_hours must be a positive number of at most "
            f"{_MAX_RECENT_DECISIONS_WINDOW_HOURS}, got {window!r}"
        )
    counts: Dict[str, int] = {}
    for key in ("max_entries", "excerpt_chars", "max_chars"):
        value = table.get(key, getattr(RecentDecisionsConfig, key))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{where} {key} must be a positive integer, got {value!r}"
            )
        counts[key] = value
    if counts["max_entries"] > _MAX_RECENT_DECISIONS_ENTRIES:
        raise ValueError(
            f"{where} max_entries must be at most "
            f"{_MAX_RECENT_DECISIONS_ENTRIES}, got {counts['max_entries']}"
        )
    floor = counts["excerpt_chars"] + _RECENT_DECISIONS_OVERHEAD_CHARS
    if counts["max_chars"] < floor:
        raise ValueError(
            f"{where} max_chars must be at least excerpt_chars + "
            f"{_RECENT_DECISIONS_OVERHEAD_CHARS} ({floor}), got {counts['max_chars']}"
        )
    return RecentDecisionsConfig(window_hours=window, **counts)


def render_recent_decisions(
    recent: "RecentDecisions", earliest: date, config: RecentDecisionsConfig
) -> Optional[str]:
    """The operational-block section for ``recent``, or ``None`` when empty.

    Newest first. Entries are dropped from the oldest end once ``max_entries``
    are shown or the next would take the section past ``max_chars``, and a
    final line counts what was dropped; an excerpt cut to ``excerpt_chars``
    ends in ``…``.
    """
    if recent.error:
        return (
            "Strategy decisions: unknown — STRATEGY.yaml could not be read at "
            f'turn start: "{_inert_text(recent.error, 200, mark_truncation=True)}"'
        )
    if not recent.decisions:
        return None
    header = (
        f"Strategy decisions since {earliest.isoformat()} "
        f"(last {config.window_hours:g}h, widened to whole days; read from "
        "STRATEGY.yaml at turn start), newest first:"
    )

    def line(decision: "RecentDecision") -> str:
        session = _inert_text(decision.session, _SESSION_CHARS, mark_truncation=True)
        label = f'session "{session}"' if session else "session not recorded"
        text = decision.decision
        if decision.rationale:
            text = f"{text} | rationale: {decision.rationale}"
        excerpt = _inert_text(text, config.excerpt_chars, mark_truncation=True)
        return f'  - {decision.day.isoformat()}, {label}: "{excerpt}"'

    def not_shown(count: int) -> str:
        return (
            f"  (+{count} older decision(s) in the window not shown: at most "
            f"{config.max_entries} entries / {config.max_chars} chars; "
            "strategy_view section=decisions lists every decision)"
        )

    lines = [line(decision) for decision in recent.decisions]
    size = len(header) + sum(1 + len(entry) for entry in lines)
    if len(lines) <= config.max_entries and size <= config.max_chars:
        return "\n".join([header, *lines])
    reserve = 1 + len(not_shown(len(lines)))
    shown: List[str] = []
    size = len(header)
    for entry in lines[: config.max_entries]:
        if size + 1 + len(entry) + reserve > config.max_chars:
            break
        shown.append(entry)
        size += 1 + len(entry)
    return "\n".join([header, *shown, not_shown(len(lines) - len(shown))])


async def _recent_decisions_section(agent: Any) -> Optional[str]:
    """Surface strategy decisions recorded in the window (#3900).

    STRATEGY.yaml reaches the prompt through the bootstrap loader, which reads
    it once and caches it, so a decision recorded in one session was not
    visible to a turn starting minutes later in another — on 2026-11-02 a wake
    turn dispatched work two minutes after another session had attached a
    safety condition to that queue. This reads the file at every turn start,
    wake and signal turns included, and shows the recent decisions as inert,
    bounded data. ``None`` when there are none or the feature is not loaded.
    """
    feat = _get_feature(agent, "StrategicMemoryFeature")
    if feat is None or not hasattr(feat, "recent_decisions"):
        return None
    try:
        config = recent_decisions_config()
    except ValueError as exc:
        logger.warning("operational_state: recent decisions not shown: %s", exc)
        return (
            "Strategy decisions: unavailable — "
            f'"{_inert_text(exc, 200, mark_truncation=True)}"'
        )
    from kestrel_sovereign.features.strategic_memory.recent_decisions import (
        earliest_recent_day,
    )

    earliest = earliest_recent_day(
        datetime.now(timezone.utc), timedelta(hours=config.window_hours)
    )
    recent = await feat.recent_decisions(earliest)
    return render_recent_decisions(recent, earliest, config)


async def build_operational_state_block(agent: Any) -> Optional[str]:
    """Always-on operational lifecycle context (#1571, #1581, #1907, #3900).

    Required operational typed events — restart_status (#1571), codex
    sandbox-approval declines (#1581), active-todo rollups (#1907) and
    recently recorded strategy decisions (#3900) — must surface in the
    agent's turn context even when the optional proactive
    ``[preturn_state]`` block is disabled. This block is minimal, DID-scoped,
    non-instructional, and silently returns ``None`` when there is nothing to
    report.
    """
    lines: List[str] = []
    for section in (
        _restart_status_section,
        _codex_decline_section,
        _active_todo_section,
        _recent_decisions_section,
    ):
        try:
            line = await section(agent)
        except Exception as exc:  # noqa: BLE001 - never break a turn
            logger.debug(
                "operational_state: %s failed: %s",
                section.__name__, exc,
            )
            continue
        if line and line.strip():
            lines.append(line)
    if not lines:
        return None
    body = "\n".join(lines)
    return f"{_OPERATIONAL_HEADER}\n{body}\n{_OPERATIONAL_FOOTER}"


async def prepare_turn_state_addendum(
    agent: Any, system_prompt_addendum: Optional[str]
) -> Optional[str]:
    """``system_prompt_addendum`` with this turn's state blocks put in front.

    The one preparation both turn entry points run, after the
    USER_PROMPT_SUBMIT hook and before context assembly: ``process_input``
    (wake, signal and API turns) and ``process_input_streaming`` (console
    turns). Before #3900 only the first ran it, so a console turn never saw
    the operational block.

    The operational block (#1571) runs for every agent; the opt-in
    ``[preturn_state]`` block (#1290) runs where it is enabled. Both ride the
    budget-aware addendum, ahead of any caller-supplied directive such as the
    constitutional canary, which stays last. Best-effort: a failure is logged
    and the addendum is returned unchanged, so it never blocks the turn.
    """
    try:
        operational = await build_operational_state_block(agent)
        if operational:
            system_prompt_addendum = (
                f"{operational}\n\n{system_prompt_addendum}"
                if system_prompt_addendum
                else operational
            )
        state = await build_preturn_state_block(agent)
        if state:
            system_prompt_addendum = (
                f"{state}\n\n{system_prompt_addendum}"
                if system_prompt_addendum
                else state
            )
    except Exception as exc:  # noqa: BLE001 - never break a turn
        logger.warning("preturn_state: injection skipped: %s", exc, exc_info=True)
    return system_prompt_addendum
