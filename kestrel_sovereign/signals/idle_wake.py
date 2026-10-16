"""The idle detector: one triage wake when the work queue drains (#3664).

On 2026-10-15 an agent's last turn ended at 23:11Z with its queue empty and
nothing running in Talon. Nothing woke it for about 16 hours. Its
continuous-work doctrine says an empty queue triggers triage, but that rule
runs inside a turn, and turns only start from wakes (a Talon completion, a
restart, a wait, an operator message). When the last item finishes and
nothing new is dispatched, no wake is pending, so the rule never runs.

This module watches for that state and emits one ``queue.drained`` COGNITION
signal (:mod:`kestrel_sovereign.signals.sources.queue_drained`) per idle
stretch. It is opt-in per agent (``[idle_wake]`` in kestrel.toml, off by
default) and runs as the ``idle_wake`` built-in cron task every five minutes.

What idle means
---------------
The agent is idle when every one of these holds:

* no Talon job is in flight: a job the ``talon`` wait provider's
  ``active_handles()`` holds that its read-only ``finished_runs()`` does not
  report finished. ``finished_runs()`` vouches that the registry was read
  whole; a finished job missing from ``active_handles()`` makes the read
  unknown. ``poll()`` is never called: it reaps and pushes;
* no wait watch is armed (on a kind whose provider is loaded: a watch nothing
  can poll never fires, and must not hold the agent quiet forever);
* no restart request the agent filed is pending, approved, updating or
  executing;
* no wake is waiting to be delivered: a wait-ledger wake in flight or parked
  until a provider-advised time, a restart wake not yet delivered, a durable
  A2A or channel cognition delivery, a scheduled ``self_followup``, or a turn
  running right now.

A state that cannot be read (an unreadable Talon registry, a Talon provider
too old to report one, a failing query, a volatile privacy mode in which
operator turns leave no record, an unreadable Hold state, a Hold) is not idle.
The detector fails closed: it does not fire, and says why in its cron result.

The wake fires once the agent has been idle for ``idle_minutes`` as observed
by the detector. It vouches only for stretches it watched: a gap between two
evaluations longer than :data:`MAX_OBSERVATION_GAP` restarts the clock. That
allowance comes from the cron's cadence, not from ``idle_minutes``: the
scheduler runs an occurrence a little late or skips one, so consecutive
evaluations are never exactly one interval apart, and a threshold of one
interval must still be reachable.

One wake per idle stretch
-------------------------
After the wake is delivered the stretch is spent. It does not fire again
until something starts new work or a person or peer turns up:

* a Talon job is dispatched (its job id appears, whether it is still running
  or already finished);
* a watch is armed (seen armed at an evaluation), or a wake is delivered
  (a delivered wake means a watch was armed or a job dispatched since);
* a restart request is filed, including one the agent files itself;
* an operator turn or an A2A turn opens.

Each of those starts a new stretch, and the idle clock starts again when the
agent next looks idle. Only new evidence counts: a job id or restart request
that was not in the previous reading, or a wake delivered later than the
newest one read. Evidence that disappears (a pruned job, a deleted row) is not
new work, and a source read for the first time is a baseline, not news. A
wake still being delivered, or a turn in progress, restarts the idle clock but
does not start a new stretch, so the triage turn's own delivery can never
re-arm it. The marker is persisted (``idle_wake_state``), so a restart neither
forgets a spent stretch nor re-fires it.

The wake is enqueued, not awaited (see :mod:`kestrel_sovereign.signals.delivery`).
A delivered or coalesced wake spends the stretch, and so does a cooperative
Stop of the triage turn, which the dispatcher reports as coalesced. A failed
one is retried on later ticks, at most :data:`MAX_WAKE_ATTEMPTS` times per
stretch, each attempt under its own durable source event id and dedupe key.

Who owns a wake
---------------
Every replica sharing the agent's database runs the cron, and the scheduler
does not pin consecutive occurrences to one of them. So the pending wake is
recorded with its owner, the process that enqueued it, and a lease the owner
renews every :data:`OWNER_HEARTBEAT_INTERVAL` while the wake's dispatch runs.
When the dispatch ends the owner settles the wake at once, so its outcome is
in the row before any replica's next evaluation. Another replica that finds
the wake pending under a live lease treats it as a turn in progress: the agent
is busy, and nothing is dispatched. Only when the lease has not been renewed
for :data:`OWNER_STALE_AFTER` (its owner stopped) is the wake recovered: it
counts as delivered when its turn stored a reply, is retried when the lookup
finds none, and stays pending while the lookup cannot be made. A process that
finds its own wake pending with no dispatch of its own behind it recovers it
the same way at once.

Every write to ``idle_wake_state`` is a compare-and-set on the row's
revision, so two evaluations racing on different replicas cannot both claim
the stretch's wake, and an evaluation working from an older reading cannot
undo a settled wake. The loser drops its write; the next evaluation reads
afresh.

Binding
-------
The wake binds to the agent's most recent chain session: the session of the
most recently delivered wait-ledger wake, resolved by the reconciler's own
rule (:func:`~kestrel_sovereign.waits.reconciler.wake_origin_session`). That
is the session the work was registered from, which may be a chat an operator
started: since #2877 a wake returns to the session that asked for the work.
With no chain session the wake carries none and the dispatcher mints a fresh
session for the turn; it never falls back to an operator's open chat.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from kestrel_sovereign.config import load_section

logger = logging.getLogger(__name__)

#: The kestrel.toml section the detector reads.
CONFIG_SECTION = "idle_wake"

#: The built-in cron task that runs one evaluation.
TASK_NAME = "idle_wake"

#: How often, in minutes, the cron task evaluates. The idle threshold is
#: measured in observed minutes, so a fire lands between ``idle_minutes`` and
#: ``idle_minutes`` plus about this interval after the agent went idle.
CRON_INTERVAL_MINUTES = 5
CRON_EXPRESSION = f"*/{CRON_INTERVAL_MINUTES} * * * *"

DEFAULT_IDLE_MINUTES = 30
#: One evaluation interval. A shorter threshold could not be observed.
MIN_IDLE_MINUTES = CRON_INTERVAL_MINUTES

#: The longest gap between two evaluations across which idle time still
#: counts as watched. It follows the cron's cadence, not ``idle_minutes``: the
#: scheduler polls every 30 seconds and runs an occurrence late under load,
#: so consecutive evaluations land a little more or less than one interval
#: apart, and a skipped occurrence makes the gap two intervals. Two intervals
#: and two minutes of delay cover both. A longer gap means the detector was not
#: running (the host was down, the scheduler stopped), and the clock restarts.
MAX_OBSERVATION_GAP = timedelta(minutes=2 * CRON_INTERVAL_MINUTES + 2)

#: Wake attempts per idle stretch before the stretch is given up.
MAX_WAKE_ATTEMPTS = 3

#: How often the process that enqueued a wake renews its lease on it while
#: the wake's dispatch runs.
OWNER_HEARTBEAT_INTERVAL = timedelta(seconds=30)
#: A lease not renewed for this long belongs to a process that stopped. It is
#: the dispatcher's runtime-owner staleness, and like it compares the owner's
#: clock with the reader's: replicas' clocks must agree to well within it.
OWNER_STALE_AFTER = timedelta(minutes=2)
#: Times an owner tries to settle its finished wake when another evaluation
#: wrote the row first. If every try loses, the owner's next evaluation
#: settles it, or a replica recovers it once the lease is stale.
SETTLE_TRIES = 3

#: A pending wake whose owner stopped (or lost its dispatch) before settling
#: it, but whose turn stored its reply: it was delivered.
REPLIED_BEFORE_RESTART = "replied_before_restart"
#: Dispatcher statuses that spend the stretch. ``coalesced`` means an
#: equivalent wake for this exact attempt was already accepted, or the triage
#: turn was stopped cooperatively.
_SPENT_STATUSES = frozenset({"ok", "coalesced", REPLIED_BEFORE_RESTART})
#: Statuses no retry can change.
_HARD_FAIL_STATUSES = frozenset({"dropped_validation", "dropped_cycle"})
#: A pending wake whose owner stopped (or lost its dispatch) and whose turn
#: stored no reply: it never finished, so it is retried.
LOST_AT_RESTART = "lost_at_restart"

#: Restart request states in which the agent is waiting on its restart.
RESTART_WORK_STATUSES = frozenset({"pending", "approved", "updating", "executing"})

#: Wake sources whose turn was opened by a peer agent.
A2A_SOURCE_PREFIX = "a2a."

#: Conversation rows read per evaluation to find newly opened turns.
CONVERSATION_SCAN_ROWS = 200
#: Delivered wait-ledger wakes searched for the most recent chain session.
CHAIN_SESSION_SCAN_ROWS = 20


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdleWakeConfig:
    """``[idle_wake]`` as it applies to one agent."""

    enabled: bool = False
    idle_minutes: int = DEFAULT_IDLE_MINUTES


def _agent_names(agent: Any) -> set:
    names = set()
    for attr in ("agent_name", "_agent_name", "did"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
    return names


def load_idle_wake_config(
    agent: Any, section: Optional[Dict[str, Any]] = None
) -> IdleWakeConfig:
    """Read ``[idle_wake]`` for ``agent``.

    Off unless ``enabled = true``. When ``agents`` lists names (or DIDs), only
    those agents are enabled; an absent or empty list enables every agent this
    host runs, as ``[preturn_state]`` does. ``idle_minutes`` defaults to
    :data:`DEFAULT_IDLE_MINUTES`; a value that is not an integer of at least
    :data:`MIN_IDLE_MINUTES` is refused with a warning and the default used.
    """
    cfg = load_section(CONFIG_SECTION) if section is None else section
    if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
        return IdleWakeConfig()
    agents = cfg.get("agents") or []
    if isinstance(agents, str):
        agents = [agents]
    if agents and not (_agent_names(agent) & {str(a).strip() for a in agents}):
        return IdleWakeConfig()
    raw = cfg.get("idle_minutes", DEFAULT_IDLE_MINUTES)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < MIN_IDLE_MINUTES:
        logger.warning(
            "[idle_wake] idle_minutes = %r is not an integer of at least %d; "
            "using %d",
            raw, MIN_IDLE_MINUTES, DEFAULT_IDLE_MINUTES,
        )
        raw = DEFAULT_IDLE_MINUTES
    return IdleWakeConfig(enabled=True, idle_minutes=raw)


# ---------------------------------------------------------------------------
# Durable state
# ---------------------------------------------------------------------------


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.warning("idle_wake_state holds an unreadable timestamp %r", value)
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class IdleWakeState:
    """One agent's ``idle_wake_state`` row.

    ``idle_since`` is the start of the idle stretch the detector is watching,
    ``None`` while the agent is not idle. ``fired_at`` is set once the
    stretch's wake is spent (delivered, or given up), and cleared only when
    new work or a new turn starts a new stretch. ``attempts`` counts the
    stretch's wake attempts and resets only with the stretch. ``evidence``
    holds what a new stretch is detected from, per source: short hashes of
    the Talon job ids and restart request ids seen, and the newest wake
    delivery time. ``conversation_cursor`` is the newest conversation row
    already examined.

    ``pending_signal_id`` is the wake enqueued and not yet settled,
    ``pending_owner`` the process that enqueued it, and
    ``pending_heartbeat_at`` when that process last renewed its lease on it.
    ``revision`` is the row's revision as read (``0``: no row yet); every
    write is conditional on it.
    """

    idle_since: Optional[str] = None
    last_evaluated_at: Optional[str] = None
    fired_at: Optional[str] = None
    fired_status: Optional[str] = None
    attempts: int = 0
    pending_signal_id: Optional[str] = None
    pending_owner: Optional[str] = None
    pending_heartbeat_at: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    conversation_cursor: Optional[int] = None
    last_wake_status: Optional[str] = None
    last_wake_error: Optional[str] = None
    revision: int = 0


class IdleWakeStateStore:
    """``idle_wake_state``: one row per agent, scoped by its DID."""

    def __init__(self, db: Any, agent_id: str) -> None:
        self._db = db
        self._agent_id = agent_id
        self._ready = False

    async def _ensure_table(self) -> None:
        if self._ready:
            return
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS idle_wake_state (
                agent_id             TEXT NOT NULL PRIMARY KEY,
                idle_since           TEXT,
                last_evaluated_at    TEXT,
                fired_at             TEXT,
                fired_status         TEXT,
                attempts             INTEGER NOT NULL DEFAULT 0,
                pending_signal_id    TEXT,
                pending_owner        TEXT,
                pending_heartbeat_at TEXT,
                evidence             TEXT,
                conversation_cursor  INTEGER,
                last_wake_status     TEXT,
                last_wake_error      TEXT,
                revision             INTEGER NOT NULL DEFAULT 0,
                updated_at           TEXT
            )
            """
        )
        self._ready = True

    async def load(self) -> IdleWakeState:
        await self._ensure_table()
        row = await self._db.fetchone(
            """
            SELECT idle_since, last_evaluated_at, fired_at, fired_status,
                   attempts, pending_signal_id, pending_owner,
                   pending_heartbeat_at, evidence, conversation_cursor,
                   last_wake_status, last_wake_error, revision
            FROM idle_wake_state WHERE agent_id = ?
            """,
            (self._agent_id,),
        )
        if not row:
            return IdleWakeState()
        evidence: Dict[str, Any] = {}
        if row[8]:
            try:
                parsed = json.loads(row[8])
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                evidence = {str(k): v for k, v in parsed.items()}
        return IdleWakeState(
            idle_since=row[0],
            last_evaluated_at=row[1],
            fired_at=row[2],
            fired_status=row[3],
            attempts=int(row[4] or 0),
            pending_signal_id=row[5],
            pending_owner=row[6],
            pending_heartbeat_at=row[7],
            evidence=evidence,
            conversation_cursor=int(row[9]) if row[9] is not None else None,
            last_wake_status=row[10],
            last_wake_error=row[11],
            revision=int(row[12] or 0),
        )

    async def save(self, state: IdleWakeState) -> Optional[IdleWakeState]:
        """Write ``state`` if the row is still at the revision it was read at.

        Returns the state as written, its revision advanced, or ``None`` when
        another writer got there first: an evaluation on another replica, or
        the owner of the pending wake renewing or settling it. That write
        stands; the caller drops its own.
        """
        await self._ensure_table()
        values = (
            state.idle_since,
            state.last_evaluated_at,
            state.fired_at,
            state.fired_status,
            int(state.attempts),
            state.pending_signal_id,
            state.pending_owner,
            state.pending_heartbeat_at,
            json.dumps(state.evidence, sort_keys=True),
            state.conversation_cursor,
            state.last_wake_status,
            state.last_wake_error,
            _iso(datetime.now(timezone.utc)),
        )
        if state.revision == 0:
            written = await self._db.execute(
                """
                INSERT INTO idle_wake_state
                    (idle_since, last_evaluated_at, fired_at, fired_status,
                     attempts, pending_signal_id, pending_owner,
                     pending_heartbeat_at, evidence, conversation_cursor,
                     last_wake_status, last_wake_error, updated_at,
                     agent_id, revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(agent_id) DO NOTHING
                """,
                values + (self._agent_id,),
            )
        else:
            written = await self._db.execute(
                """
                UPDATE idle_wake_state SET
                    idle_since = ?, last_evaluated_at = ?, fired_at = ?,
                    fired_status = ?, attempts = ?, pending_signal_id = ?,
                    pending_owner = ?, pending_heartbeat_at = ?, evidence = ?,
                    conversation_cursor = ?, last_wake_status = ?,
                    last_wake_error = ?, updated_at = ?,
                    revision = revision + 1
                WHERE agent_id = ? AND revision = ?
                """,
                values + (self._agent_id, state.revision),
            )
        if written != 1:
            return None
        return replace(state, revision=state.revision + 1)

    async def renew(self, signal_id: str, owner: str, heartbeat_at: str) -> bool:
        """Renew ``owner``'s lease on the pending wake ``signal_id``.

        ``False`` when the wake is no longer pending under that owner: it was
        settled, or recovered elsewhere after the lease went stale. A renewal
        advances the revision, so an evaluation that read the lease before it
        cannot then act on the older reading. One that repeats the stored
        heartbeat changes nothing and leaves the revision alone.
        """
        await self._ensure_table()
        written = await self._db.execute(
            """
            UPDATE idle_wake_state SET
                revision = CASE WHEN pending_heartbeat_at = ?
                                THEN revision ELSE revision + 1 END,
                pending_heartbeat_at = ?
            WHERE agent_id = ? AND pending_signal_id = ? AND pending_owner = ?
            """,
            (heartbeat_at, heartbeat_at, self._agent_id, signal_id, owner),
        )
        return written == 1


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """What one evaluation saw.

    ``work`` is work in progress (a stretch ends), ``deliveries`` a wake still
    on its way (the clock restarts, the stretch goes on), ``unknown`` a state
    that could not be read (fail closed). ``evidence`` maps each evidence
    source to what was read from it now, and omits a source that was not read.
    ``activity`` names what started since the last evaluation.
    """

    work: List[str] = field(default_factory=list)
    deliveries: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    activity: List[str] = field(default_factory=list)
    conversation_cursor: Optional[int] = None

    @property
    def idle(self) -> bool:
        return not (self.work or self.deliveries or self.unknown)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _short_ids(items) -> List[str]:
    """Sorted short hashes of ``items``: enough to tell a new id from a seen
    one without storing every id verbatim."""
    return sorted(
        {hashlib.sha256(str(item).encode("utf-8")).hexdigest()[:12] for item in items}
    )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _dispatch_outcome(handle: Any) -> Tuple[str, Optional[str]]:
    """The settled status of a finished wake dispatch, and its error."""
    try:
        result = handle.task.result()
    except asyncio.CancelledError:
        return "dispatch_cancelled", None
    except Exception as exc:  # noqa: BLE001 - the dispatch raised
        return "dispatcher_raised", _describe(exc)
    status = getattr(getattr(result, "status", None), "value", None)
    if not isinstance(status, str):
        return "tooling_error", f"dispatch returned no usable status: {result!r}"
    return status, getattr(result, "error", None) or None


class IdleWakeDetector:
    """One agent's idle detector, held on ``agent._idle_wake_detector``.

    It owns the wakes it enqueues: it keeps their in-memory dispatch handles
    and, while each dispatch runs, the task that renews its lease and settles
    it when the dispatch ends.
    """

    def __init__(
        self,
        agent: Any,
        *,
        clock: Callable[[], datetime] = _utc_now,
        heartbeat_interval: timedelta = OWNER_HEARTBEAT_INTERVAL,
    ) -> None:
        from kestrel_sovereign.features.storage_access import (
            resolve_feature_database,
            resolve_scoped_agent_did,
        )
        from kestrel_sovereign.storage.async_wait_signal_store import (
            WaitSignalStore,
        )

        self._agent = agent
        self._clock = clock
        # Every read below is scoped to this DID; a missing identity refuses
        # construction rather than reading another agent's rows (#3251).
        self._agent_id = resolve_scoped_agent_did(agent)
        self._db = resolve_feature_database(agent)
        if self._db is None:
            raise RuntimeError("idle wake: the agent has no database")
        self._state_store = IdleWakeStateStore(self._db, self._agent_id)
        self._wait_store = WaitSignalStore(self._db, self._agent_id)
        # This process, as the owner of the wakes it enqueues. A restarted
        # process is a new owner: it cannot renew the old one's lease.
        self._owner = f"idle-wake:{secrets.token_urlsafe(12)}"
        self._heartbeat_seconds = heartbeat_interval.total_seconds()
        self._pending: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------

    async def tick(
        self,
        config: IdleWakeConfig,
        *,
        pending_followups: Optional[Callable[[], Awaitable[int]]] = None,
    ) -> Dict[str, Any]:
        """Run one evaluation and return a JSON-able report of it."""
        if not config.enabled:
            return {"skipped": "disabled", "fired": False}
        async with self._lock:
            return await self._tick(config, pending_followups)

    async def _tick(
        self,
        config: IdleWakeConfig,
        pending_followups: Optional[Callable[[], Awaitable[int]]],
    ) -> Dict[str, Any]:
        now = self._clock()
        state = await self._state_store.load()
        report: Dict[str, Any] = {
            "idle_minutes": config.idle_minutes,
            "fired": False,
        }

        # --- A wake enqueued earlier and not settled yet ----------------
        if state.pending_signal_id:
            signal_id = state.pending_signal_id
            ours = state.pending_owner == self._owner
            handle = self._pending.get(signal_id) if ours else None
            if handle is not None and not handle.task.done():
                # The triage turn is running here: the agent is busy, and
                # the stretch's own wake must not start a new one.
                return await self._wake_in_flight(state, now, report, "wake_in_flight")
            if not ours and not self._lease_stale(state, now):
                # Another live process enqueued it, and its triage turn may
                # be running there. Only that process can say how it ended.
                return await self._wake_in_flight(
                    state, now, report, "wake_in_flight_elsewhere"
                )
            if handle is not None:
                # Settling it when the dispatch ended failed; settle it now.
                status, error = _dispatch_outcome(handle)
            else:
                # Its owner stopped, or this process lost its dispatch.
                status, error = await self._reply_outcome(signal_id)
            if status is None:
                return await self._commit(
                    replace(state, idle_since=None, last_evaluated_at=_iso(now)),
                    report,
                    "unknown",
                    unknown=[error or "undecided"],
                )
            # Durable before anything else can fail: a harvested result that
            # was never written would be read back as a wake lost.
            settled = await self._state_store.save(
                self._settle(state, status, error, now)
            )
            if settled is None:
                report["verdict"] = "state_conflict"
                return report
            state = settled
            self._pending.pop(signal_id, None)
            report["harvested"] = {"status": status, "wake_error": error or ""}

        observation = await self._observe(state, pending_followups)
        report.update(
            work=observation.work,
            deliveries=observation.deliveries,
            unknown=observation.unknown,
            activity=observation.activity,
        )

        # --- Transition --------------------------------------------------
        evidence = dict(state.evidence)
        evidence.update(observation.evidence)
        idle_since = _parse(state.idle_since)
        fired_at = state.fired_at
        fired_status = state.fired_status
        attempts = state.attempts
        if observation.work or observation.activity:
            # New work or a new turn: this stretch is over.
            fired_at = None
            fired_status = None
            attempts = 0
            idle_since = None
        if not observation.idle:
            idle_since = None
        else:
            last_evaluated = _parse(state.last_evaluated_at)
            unobserved = (
                last_evaluated is None or now - last_evaluated > MAX_OBSERVATION_GAP
            )
            if idle_since is None or unobserved:
                # The clock starts now: the detector vouches only for idle
                # time it watched. The stretch, and its attempt count, go on.
                idle_since = now
        state = replace(
            state,
            idle_since=_iso(idle_since) if idle_since is not None else None,
            last_evaluated_at=_iso(now),
            fired_at=fired_at,
            fired_status=fired_status,
            attempts=attempts,
            evidence=evidence,
            conversation_cursor=(
                observation.conversation_cursor
                if observation.conversation_cursor is not None
                else state.conversation_cursor
            ),
        )
        report.update(idle_since=state.idle_since, fired_at=state.fired_at)

        if not observation.idle:
            busy = observation.work or observation.deliveries
            return await self._commit(state, report, "busy" if busy else "unknown")
        if state.fired_at is not None:
            return await self._commit(state, report, "already_fired")
        idle_for = now - idle_since
        if idle_for < timedelta(minutes=config.idle_minutes):
            return await self._commit(
                state,
                report,
                "idle",
                idle_for_minutes=int(idle_for.total_seconds() // 60),
            )
        return await self._fire(state, config, idle_since, now, report)

    async def _commit(
        self,
        state: IdleWakeState,
        report: Dict[str, Any],
        verdict: str,
        **details: Any,
    ) -> Dict[str, Any]:
        """Write ``state`` and finish ``report`` with ``verdict``.

        A lost compare-and-set reports ``state_conflict`` instead: another
        evaluation, or the owner of the pending wake, wrote the row since it
        was read, and its write stands. The next evaluation reads afresh.
        """
        if await self._state_store.save(state) is None:
            report.update(verdict="state_conflict", superseded=verdict)
            return report
        report.update(verdict=verdict, **details)
        return report

    async def _wake_in_flight(
        self,
        state: IdleWakeState,
        now: datetime,
        report: Dict[str, Any],
        verdict: str,
    ) -> Dict[str, Any]:
        """The agent is busy with the stretch's own wake. The idle clock
        restarts; the stretch, and its attempt count, go on."""
        return await self._commit(
            replace(state, idle_since=None, last_evaluated_at=_iso(now)),
            report,
            verdict,
            signal_id=state.pending_signal_id,
            lease_heartbeat_at=state.pending_heartbeat_at,
        )

    @staticmethod
    def _lease_stale(state: IdleWakeState, now: datetime) -> bool:
        """Whether the pending wake's owner has stopped renewing its lease."""
        heartbeat = _parse(state.pending_heartbeat_at)
        return heartbeat is None or now - heartbeat > OWNER_STALE_AFTER

    # ------------------------------------------------------------------

    async def _fire(
        self,
        state: IdleWakeState,
        config: IdleWakeConfig,
        idle_since: datetime,
        now: datetime,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        from kestrel_sovereign.signals.sources.queue_drained import (
            build_queue_drained_signal,
        )

        dispatcher = getattr(self._agent, "dispatcher", None)
        if dispatcher is None or not hasattr(dispatcher, "enqueue_signal"):
            return await self._commit(state, report, "no_dispatcher")

        attempt = state.attempts + 1
        session_id = await self._chain_session()
        period = _iso(idle_since)
        signal = build_queue_drained_signal(
            target_agent=self._agent_id,
            idle_since=period,
            idle_minutes=config.idle_minutes,
            observed_idle_minutes=int((now - idle_since).total_seconds() // 60),
            attempt=attempt,
            session_id=session_id,
        )
        # The claim, durable before the enqueue: a crash between the two is a
        # lost wake that is recovered and retried, never an unrecorded one.
        # Its compare-and-set is what keeps two replicas from both firing.
        claimed = await self._state_store.save(
            replace(
                state,
                attempts=attempt,
                pending_signal_id=signal.id,
                pending_owner=self._owner,
                pending_heartbeat_at=_iso(now),
            )
        )
        if claimed is None:
            report.update(verdict="state_conflict", superseded="fired")
            return report
        state = claimed
        try:
            handle = await dispatcher.enqueue_signal(
                signal, source_event_id=f"idle:{period}#{attempt}"
            )
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            logger.warning("idle wake: enqueue_signal raised: %s", exc)
            settled = self._settle(state, "dispatcher_raised", _describe(exc), now)
            report.update(wake_error=_describe(exc), fired_at=settled.fired_at)
            return await self._commit(settled, report, "dispatch_error")
        self._pending[signal.id] = handle
        self._agent._track_background_task(
            self._keep_lease(signal.id, handle),
            name=f"idle_wake_lease:{signal.id}",
        )
        logger.info(
            "idle wake: agent idle since %s; enqueued queue.drained %s "
            "(attempt %d, session %s)",
            period, signal.id, attempt, session_id or "minted",
        )
        report.update(
            verdict="fired",
            fired=True,
            signal_id=signal.id,
            attempt=attempt,
            session_bound=bool(session_id),
        )
        return report

    async def _keep_lease(self, signal_id: str, handle: Any) -> None:
        """Renew this process's lease on its wake while the dispatch runs,
        then settle the wake.

        The lease is what tells another replica that the triage turn may
        still be running. A renewal that fails is retried at the next
        interval rather than ending the watch: a lease left to go stale while
        the turn runs would let another replica dispatch the wake again.
        """
        while not handle.task.done():
            await asyncio.wait({handle.task}, timeout=self._heartbeat_seconds)
            if handle.task.done():
                break
            try:
                held = await self._state_store.renew(
                    signal_id, self._owner, _iso(self._clock())
                )
            except Exception as exc:  # noqa: BLE001 - retried next interval
                logger.warning(
                    "idle wake: could not renew the lease on %s: %s",
                    signal_id, _describe(exc),
                )
                continue
            if not held:
                # Recovered elsewhere after the lease went stale, although
                # the dispatch is still running here.
                logger.warning(
                    "idle wake: lost the lease on %s while its dispatch runs; "
                    "another replica may have dispatched the wake again",
                    signal_id,
                )
                self._pending.pop(signal_id, None)
                return
        async with self._lock:
            await self._settle_own(signal_id, handle)

    async def _settle_own(self, signal_id: str, handle: Any) -> None:
        """Record the outcome of this process's own finished wake."""
        status, error = _dispatch_outcome(handle)
        for _ in range(SETTLE_TRIES):
            try:
                state = await self._state_store.load()
                if (
                    state.pending_signal_id != signal_id
                    or state.pending_owner != self._owner
                ):
                    break  # already settled, or recovered elsewhere
                settled = self._settle(state, status, error, self._clock())
                if await self._state_store.save(settled) is not None:
                    break
            except Exception as exc:  # noqa: BLE001 - an evaluation settles it
                logger.warning(
                    "idle wake: could not settle %s (%s); the next evaluation "
                    "will: %s", signal_id, status, _describe(exc),
                )
                return
        else:
            logger.warning(
                "idle wake: could not settle %s (%s) in %d tries; the next "
                "evaluation will", signal_id, status, SETTLE_TRIES,
            )
            return
        self._pending.pop(signal_id, None)

    async def _reply_outcome(
        self, signal_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """How a wake whose dispatch no live process holds ended.

        Its turn either stored the reply stamped with this wake's id (#3655)
        or never finished; only the second is retried. ``None`` means
        undecided: the reply could not be looked up. That is neither
        delivered nor lost, so the wake stays pending and the next evaluation
        asks again.
        """
        from kestrel_sovereign.features.storage_access import (
            resolve_conversation_store,
        )

        store = resolve_conversation_store(self._agent)
        if store is None:
            return None, "conversation_store_unavailable"
        try:
            reply_id = await store.find_signal_wake_reply_id(signal_id)
        except Exception as exc:  # noqa: BLE001 - undecided, not lost
            return None, f"reply_lookup_failed: {_describe(exc)}"
        if reply_id is not None:
            return REPLIED_BEFORE_RESTART, None
        return LOST_AT_RESTART, None

    def _settle(
        self,
        state: IdleWakeState,
        status: str,
        error: Optional[str],
        now: datetime,
    ) -> IdleWakeState:
        """Record a harvested wake and decide whether the stretch is spent."""
        fired_at = state.fired_at
        fired_status = state.fired_status
        if status in _SPENT_STATUSES:
            fired_at, fired_status = _iso(now), status
        elif status in _HARD_FAIL_STATUSES or state.attempts >= MAX_WAKE_ATTEMPTS:
            fired_at, fired_status = _iso(now), f"gave_up:{status}"
            logger.warning(
                "idle wake: giving up on this idle stretch after %d attempt(s); "
                "last status %s%s",
                state.attempts, status, f" ({error})" if error else "",
            )
        else:
            logger.warning(
                "idle wake: wake attempt %d not delivered (%s%s); retrying on "
                "a later tick",
                state.attempts, status, f": {error}" if error else "",
            )
        return replace(
            state,
            pending_signal_id=None,
            pending_owner=None,
            pending_heartbeat_at=None,
            fired_at=fired_at,
            fired_status=fired_status,
            last_wake_status=status,
            last_wake_error=error,
        )

    # ------------------------------------------------------------------

    async def _observe(
        self,
        state: IdleWakeState,
        pending_followups: Optional[Callable[[], Awaitable[int]]],
    ) -> Observation:
        obs = Observation()
        await self._observe_privacy_and_hold(obs)
        await self._observe_conversation(state, obs)
        await self._observe_wait_ledger(obs)
        await self._observe_restarts(obs)
        await self._observe_durable_deliveries(obs)
        live_turn = getattr(self._agent, "_live_turn_id", None)
        if isinstance(live_turn, str) and live_turn:
            obs.deliveries.append("turn_in_progress")
        if pending_followups is not None:
            try:
                count = await pending_followups()
            except Exception as exc:  # noqa: BLE001 - fail closed
                obs.unknown.append(f"self_followup_unreadable: {_describe(exc)}")
            else:
                if count:
                    obs.deliveries.append(f"self_followup_scheduled:{count}")
        # The Talon read is the expensive one (it reads every finished job's
        # log). Skip it when the agent is already known not to be idle; its
        # evidence then stays where it was and is compared next time.
        if obs.idle:
            await self._observe_talon(obs)
        self._detect_activity(state, obs)
        return obs

    async def _observe_privacy_and_hold(self, obs: Observation) -> None:
        from kestrel_sovereign.features.storage_access import (
            hides_persisted_user_content,
        )
        from kestrel_sovereign.hold.enforcement import get_effective_hold_state

        if hides_persisted_user_content(self._agent):
            # Operator turns leave no record in a volatile mode, so a new
            # stretch could not be told from the old one.
            obs.unknown.append("volatile_privacy_mode")
        try:
            hold = await get_effective_hold_state(self._agent)
        except Exception as exc:  # noqa: BLE001 - fail closed
            obs.unknown.append(f"hold_state_unreadable: {_describe(exc)}")
        else:
            if hold is not None and hold.held:
                obs.unknown.append("held")

    async def _observe_conversation(
        self, state: IdleWakeState, obs: Observation
    ) -> None:
        from kestrel_sovereign.features.storage_access import (
            resolve_conversation_store,
        )
        from kestrel_sovereign.storage.session_grouping import (
            is_human_turn,
            opens_turn,
            signal_wake_source,
        )

        store = resolve_conversation_store(self._agent)
        if store is None:
            obs.unknown.append("conversation_store_unavailable")
            return
        try:
            if state.conversation_cursor is None:
                # First look: everything so far is the baseline.
                obs.conversation_cursor = await store.latest_row_id() or 0
                return
            rows = await store.user_rows_after(
                state.conversation_cursor, limit=CONVERSATION_SCAN_ROWS
            )
        except Exception as exc:  # noqa: BLE001 - fail closed
            obs.unknown.append(f"conversation_unreadable: {_describe(exc)}")
            return
        for row_id, metadata in rows:
            obs.conversation_cursor = row_id
            if not opens_turn("user", metadata):
                continue
            if is_human_turn("user", metadata):
                obs.activity.append(f"operator_turn:{row_id}")
                continue
            source = signal_wake_source(metadata) or ""
            if source.startswith(A2A_SOURCE_PREFIX):
                obs.activity.append(f"peer_turn:{row_id}")

    async def _observe_wait_ledger(self, obs: Observation) -> None:
        registry = getattr(self._agent, "wait_registry", None)
        if registry is None:
            # Without the registry no watch can be judged pollable.
            obs.unknown.append("wait_registry_unavailable")
            return

        def loaded(kind: str) -> bool:
            return registry.get(kind) is not None

        try:
            watched = await self._wait_store.list_watched()
            pending = await self._wait_store.list_pending()
            parked = await self._wait_store.list_parked()
            latest = await self._wait_store.list_latest_delivered(limit=1)
        except Exception as exc:  # noqa: BLE001 - fail closed
            obs.unknown.append(f"wait_ledger_unreadable: {_describe(exc)}")
            return
        obs.work.extend(
            f"watch_armed:{s.kind}:{s.handle}" for s in watched if loaded(s.kind)
        )
        obs.deliveries.extend(f"wake_in_flight:{s.kind}:{s.handle}" for s in pending)
        obs.deliveries.extend(
            f"wake_parked:{s.kind}:{s.handle}" for s in parked if loaded(s.kind)
        )
        delivered = latest[0].delivered_at_utc() if latest else None
        obs.evidence["wait"] = _iso(delivered) if delivered is not None else ""

    async def _observe_restarts(self, obs: Observation) -> None:
        features = getattr(self._agent, "features", None)
        if not isinstance(features, dict) or "RestartCoordinatorFeature" not in features:
            # No coordinator, no requests: nothing to wait on, nothing read.
            return
        from kestrel_sovereign.features.restart_coordinator.store import (
            list_requests,
            list_requests_needing_wake,
        )

        try:
            requests = await list_requests(self._db, agent_id=self._agent_id)
            needing_wake = await list_requests_needing_wake(
                self._db, agent_id=self._agent_id
            )
        except Exception as exc:  # noqa: BLE001 - fail closed
            obs.unknown.append(f"restart_requests_unreadable: {_describe(exc)}")
            return
        obs.work.extend(
            f"restart_{r.status}:{r.id}"
            for r in requests
            if r.status in RESTART_WORK_STATUSES
        )
        obs.deliveries.extend(
            f"restart_wake_undelivered:{r.id}"
            for r in needing_wake
            if r.status not in RESTART_WORK_STATUSES
        )
        obs.evidence["restart"] = _short_ids(r.id for r in requests)

    async def _observe_durable_deliveries(self, obs: Observation) -> None:
        from kestrel_sovereign.signals.durable import (
            INITIAL_RESERVED,
            LEASED,
            PENDING,
            RETRY,
        )
        from kestrel_sovereign.signals.sources.a2a import (
            DURABLE_COGNITION_CONSUMER_ID as A2A_COMPLETE_CONSUMER,
        )
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            DURABLE_COGNITION_CONSUMER_ID as A2A_SUBMITTED_CONSUMER,
        )
        from kestrel_sovereign.signals.sources.channels import (
            DURABLE_COGNITION_CONSUMER_ID as CHANNEL_CONSUMER,
        )

        dispatcher = getattr(self._agent, "dispatcher", None)
        listing = getattr(dispatcher, "list_durable_deliveries", None)
        if not callable(listing):
            obs.unknown.append("durable_deliveries_unobservable")
            return
        for consumer_id in (
            A2A_COMPLETE_CONSUMER,
            A2A_SUBMITTED_CONSUMER,
            CHANNEL_CONSUMER,
        ):
            try:
                rows = await listing(
                    consumer_id=consumer_id,
                    statuses=[PENDING, RETRY, INITIAL_RESERVED, LEASED],
                    limit=1,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed
                obs.unknown.append(
                    f"durable_deliveries_unreadable:{consumer_id}: {_describe(exc)}"
                )
                continue
            if rows:
                obs.deliveries.append(f"durable_delivery_pending:{consumer_id}")

    async def _observe_talon(self, obs: Observation) -> None:
        from kestrel_sovereign.features.strategic_memory.run_history import (
            FINISHED_RUNS_METHOD,
            FINISHED_RUNS_REQUIREMENT,
            TALON_WAIT_KIND,
        )

        registry = getattr(self._agent, "wait_registry", None)
        provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
        if provider is None:
            # No Talon on this agent: no jobs, nothing read.
            return
        read = getattr(provider, FINISHED_RUNS_METHOD, None)
        if not callable(read):
            # ``active_handles()`` alone cannot tell an empty registry from
            # an unreadable one (see run_history), so there is no safe read.
            obs.unknown.append(
                f"talon_provider_outdated: requires {FINISHED_RUNS_REQUIREMENT}"
            )
            return
        try:
            report = await read()
            if not isinstance(report, dict) or report.get("complete") is not True:
                reason = report.get("reason") if isinstance(report, dict) else None
                obs.unknown.append(
                    f"talon_registry_unreadable: {reason or 'incomplete'}"
                )
                return
            runs = report.get("runs")
            if not isinstance(runs, list):
                obs.unknown.append("talon_registry_unreadable: no list of runs")
                return
            finished = {
                str(run["job_id"])
                for run in runs
                if isinstance(run, dict) and isinstance(run.get("job_id"), str)
            }
            # Every job the coordinator holds, terminal or not; the ones it
            # did not report finished are still running.
            known = {str(h) for h in await provider.active_handles()}
        except Exception as exc:  # noqa: BLE001 - fail closed
            obs.unknown.append(f"talon_registry_unreadable: {_describe(exc)}")
            return
        if finished - known:
            # ``active_handles()`` answers from memory and cannot say it failed
            # to read the registry. Missing a job the registry reported
            # finished means it may be missing running ones too.
            obs.unknown.append("talon_registry_inconsistent")
            return
        obs.work.extend(f"talon_job:{job}" for job in sorted(known - finished))
        obs.evidence["talon"] = _short_ids(known | finished)

    @staticmethod
    def _detect_activity(state: IdleWakeState, obs: Observation) -> None:
        """Name the new work since the last evaluation, and settle what to keep.

        Only additions count: a Talon job id or restart request id not seen
        before, or a wake delivered later than the newest one seen. Evidence
        that disappears (a pruned job, a deleted row) is not new work. A source
        read for the first time sets its baseline and is not news.
        """
        for source, name in (
            ("talon", "talon_job_dispatched"),
            ("restart", "restart_filed"),
        ):
            current = obs.evidence.get(source)
            previous = state.evidence.get(source)
            if isinstance(current, list) and isinstance(previous, list):
                if set(current) - set(previous):
                    obs.activity.append(name)
        if "wait" in obs.evidence:
            current = _parse(obs.evidence["wait"])
            # "" is a read that found no delivery yet: the first delivery
            # after it is new. A missing key was never read: a baseline.
            seen = "wait" in state.evidence
            previous = _parse(state.evidence.get("wait"))
            # A deleted newest row moves ``current`` back, which is not newer.
            # Any wake delivered after a fire was polled after it, so it is
            # newer than whatever was read before.
            newer = current is not None and (previous is None or current > previous)
            if seen and newer:
                obs.activity.append("wake_delivered")

    async def _chain_session(self) -> Optional[str]:
        """The agent's most recent chain session, or ``None``.

        The newest delivered wait-ledger wake that resolves to a session,
        through the reconciler's own rule. ``None`` lets the dispatcher mint a
        fresh session; there is no fallback to an operator's open one.
        """
        from kestrel_sovereign.waits.reconciler import wake_origin_session

        registry = getattr(self._agent, "wait_registry", None)
        try:
            rows = await self._wait_store.list_latest_delivered(
                limit=CHAIN_SESSION_SCAN_ROWS
            )
            for row in rows:
                provider = registry.get(row.kind) if registry is not None else None
                session = await wake_origin_session(provider, row.handle, row)
                if session:
                    return session
        except Exception as exc:  # noqa: BLE001 - an unbound wake still wakes
            logger.warning(
                "idle wake: could not resolve the chain session; the wake "
                "starts a fresh one: %s", exc,
            )
        return None


async def run_idle_wake(
    agent: Any,
    *,
    config: Optional[IdleWakeConfig] = None,
    pending_followups: Optional[Callable[[], Awaitable[int]]] = None,
) -> Dict[str, Any]:
    """Run one idle evaluation for ``agent``: the ``idle_wake`` cron handler.

    Builds and caches the agent's detector on ``agent._idle_wake_detector``.
    ``pending_followups`` counts the agent's scheduled ``self_followup``
    wakes; the scheduler owns that table, so it supplies the count.
    """
    if config is None:
        config = load_idle_wake_config(agent)
    if not config.enabled:
        return {"skipped": "disabled", "fired": False}
    detector = getattr(agent, "_idle_wake_detector", None)
    if detector is None:
        detector = IdleWakeDetector(agent)
        agent._idle_wake_detector = detector
    return await detector.tick(config, pending_followups=pending_followups)
