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

The block is merged into ``system_prompt_addendum`` in
``_process_input_traced_locked`` (after the USER_PROMPT_SUBMIT hook,
before context assembly) so it rides the existing, budget-aware addendum
plumbing rather than adding a parallel injection path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

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
        from kestrel_sovereign.a2a.types import TaskState
        tasks = await task_manager.task_store.list_tasks(limit=50)
    except Exception:
        return None
    pending = [
        t for t in tasks
        if t.status.state in (TaskState.SUBMITTED, TaskState.WORKING)
    ]
    n = len(pending)
    if n == 0:
        return "A2A inbox: no pending tasks"
    lines = [f"A2A inbox: {n} pending task(s); last 3:"]
    for task in pending[-3:][::-1]:
        meta = task.metadata or {}
        sender = str(meta.get("sender") or "?")
        skill = str(meta.get("skill") or "")
        first_text = ""
        if task.history:
            for part in task.history[0].parts:
                if hasattr(part, "text"):
                    first_text = part.text
                    break
        first_text = first_text.replace("\n", " ").strip()
        label = f"{skill or 'task'} from {sender}"
        lines.append(
            f"  - [{task.status.state.value}] {label}: {first_text[:90]}"
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
    agent_did = getattr(agent, "did", None) or getattr(
        agent, "_did", None,
    )
    if db is None or not agent_did:
        return None
    try:
        from kestrel_sovereign.features.restart_coordinator.event_store import (
            list_recent_events_for_agent_context,
        )
        rows = await list_recent_events_for_agent_context(
            db, agent_id=str(agent_did), limit=20,
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
    agent_id = getattr(agent, "did", None) or getattr(
        agent, "_did", None,
    )
    if db is None or not agent_id:
        return None
    try:
        from kestrel_sovereign.llm.codex_decline_events import (
            list_recent_declines_for_agent,
        )
        rows = await list_recent_declines_for_agent(
            db, agent_id=str(agent_id), limit=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "operational_state: codex_decline read failed: %s", exc,
        )
        return None
    if not rows:
        return None
    by_reason: dict[str, int] = {}
    for ev in rows:
        key = ev.reason.split(":", 1)[0]
        by_reason[key] = by_reason.get(key, 0) + 1
    counts = ", ".join(
        f"{n} {r}" for r, n in sorted(by_reason.items())
    )
    latest = rows[0]
    latest_tool = latest.tool[:80]
    return (
        f"Codex declines (recent {len(rows)}): {counts}; "
        f"latest: {latest.request} → {latest_tool!r}"
    )


async def build_operational_state_block(agent: Any) -> Optional[str]:
    """Always-on operational lifecycle context (#1571, #1581).

    Required operational typed events — restart_status (#1571) plus
    codex sandbox-approval declines (#1581) — must surface in the
    agent's turn context even when the optional proactive
    ``[preturn_state]`` block is disabled. This block is minimal,
    DID-scoped, non-instructional, and silently returns ``None``
    when there is nothing to report.
    """
    lines: List[str] = []
    for section in (_restart_status_section, _codex_decline_section):
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
