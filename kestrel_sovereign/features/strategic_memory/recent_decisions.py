"""Which STRATEGY.yaml decisions are recent enough to show at every turn (#3900).

STRATEGY.yaml reaches the system prompt through the bootstrap loader, which
reads it once and caches it until a reload. A decision recorded in one session
is therefore not in the prompt of a turn that starts minutes later in another
session of the same agent: on 2026-11-02 a wake turn dispatched a work order
two minutes after a decision in another session had attached a safety
condition to that queue, and showed no sign of having seen it. The always-on
operational block reads the file again at the start of every turn and shows
the decisions selected here.

Decisions record a calendar day, not an instant: ``strategy_add_decision``
writes ``date.today()``, the host's local date. A window in hours is therefore
widened to whole days, never narrowed: a decision is recent when its day is no
earlier than the local day the window starts on. A 24-hour window thus shows
today's and yesterday's decisions, because a decision dated yesterday may have
been written a minute before midnight. A decision whose day cannot be read is
never recent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

from .timestamps import parse_instant


@dataclass(frozen=True)
class RecentDecision:
    """One decision as STRATEGY.yaml holds it, reduced to what a turn shows."""

    day: date
    session: str
    decision: str
    rationale: str


@dataclass(frozen=True)
class RecentDecisions:
    """The recent decisions, newest first, or why they could not be read.

    ``error`` is set when STRATEGY.yaml exists but could not be read at turn
    start: "no recent decisions" and "could not tell" are different facts, and
    a turn must not be told the first when only the second is true.
    """

    decisions: Tuple[RecentDecision, ...] = ()
    error: Optional[str] = None


def decision_day(value: Any) -> Optional[date]:
    """The local calendar day a decision's ``date`` names, or ``None``.

    ``strategy_add_decision`` writes ``YYYY-MM-DD`` text; a hand edit may leave
    an unquoted date, which YAML loads as a ``date``, or a full timestamp. A
    timestamp is converted to the host's local day, the day ``date.today()``
    would have written for it; one without an offset is UTC, as YAML and
    :func:`parse_instant` both read it. A timestamp with no local day (year 1
    west of UTC) is unreadable, never an error.
    """
    if isinstance(value, datetime):
        instant: Optional[datetime] = value
    elif isinstance(value, date):
        return value
    elif not isinstance(value, str):
        return None
    else:
        text = value.strip()
        if len(text) == 10:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
        instant = parse_instant(text)
    if instant is None:
        return None
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    try:
        return instant.astimezone().date()
    except (OverflowError, OSError, ValueError):
        return None


def earliest_recent_day(now: datetime, window: timedelta) -> date:
    """The local day of the instant ``window`` before ``now``.

    Subtracted as elapsed time and only then converted to the local day, so a
    daylight-saving change inside the window cannot move the boundary.
    """
    return (now - window).astimezone().date()


def decisions_since(
    entries: Iterable[Dict[str, Any]], earliest: date
) -> Tuple[RecentDecision, ...]:
    """Decisions dated ``earliest`` or later, newest first.

    Entries are appended to STRATEGY.yaml in the order they were recorded, so
    among decisions of one day the later entry is the newer one. Entries with
    no readable day or no decision text are left out.
    """
    selected = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        day = decision_day(entry.get("date"))
        text = str(entry.get("decision") or "").strip()
        if day is None or day < earliest or not text:
            continue
        selected.append(
            (
                day,
                position,
                RecentDecision(
                    day=day,
                    session=str(entry.get("session") or "").strip(),
                    decision=text,
                    rationale=str(entry.get("rationale") or "").strip(),
                ),
            )
        )
    selected.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(decision for _day, _position, decision in selected)
