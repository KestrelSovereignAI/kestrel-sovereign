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

* no Talon job is in flight in the agent's Talon lane: the lane's ``talon``
  wait provider's read-only ``finished_runs()`` reports the registry read
  whole (``complete``) and lists no job in its ``unfinished`` key. Both come
  from one read, so a ``complete`` report vouches for that list. A report
  without the key is unknown: kestrel-feature-talon reports unfinished jobs
  from 0.2.14 (``UNFINISHED_JOBS_REQUIREMENT``), and through an older
  provider the detector never fires. ``active_handles()`` is never the list
  of running jobs: it reads the registry a second time and answers with what
  it remembers when it cannot. ``poll()`` is never called: it reaps and
  pushes;
* no wait watch is armed (on a kind whose provider is loaded: a watch nothing
  can poll never fires, and must not hold the agent quiet forever);
* no restart request the agent filed is pending, approved, updating or
  executing;
* no wake is waiting to be delivered: a wait-ledger wake in flight or parked
  until a provider-advised time, a restart wake not yet delivered, a durable
  A2A or channel cognition delivery, a ``self_followup`` the scheduler has not
  finished (scheduled, or claimed and awaiting its turn or crash recovery), or
  a turn running right now.

A state that cannot be read (an unreadable Talon registry, a Talon provider
that does not report its unfinished jobs, a Talon dispatcher that is not
running on this host, a failing query, a volatile privacy mode in which
operator turns leave no record, an unreadable Hold state, a Hold) is not
idle. The detector fails closed: it does not fire, and says why in its cron
result.

The wake fires once the agent has been idle for ``idle_minutes`` as observed
by the detector. It vouches only for stretches it watched: a gap between two
evaluations longer than :data:`MAX_OBSERVATION_GAP` restarts the clock. That
allowance comes from the cron's cadence, not from ``idle_minutes``: the
scheduler runs an occurrence a little late or skips one, so consecutive
evaluations are never exactly one interval apart, and a threshold of one
interval must still be reachable.

Whose Talon registry
--------------------
An agent's Talon work does not always run on that agent. A host can name one
agent as its only Talon dispatcher (``[talon.policy] dispatcher`` in
kestrel-feature-talon 0.2.14): every other agent sends that agent a work
order (an A2A task) and the dispatcher's Talon runs it. The selecting
agent's own registry then stays empty while the dispatcher's lane is busy
(#3758), so reading it would report Talon idle mid-run. The detector reads
the dispatcher's registry instead, and never the selecting agent's own. It
takes the dispatcher from ``[talon.policy] dispatcher`` in its kestrel.toml,
the file a fleet host's Talon ``config_path`` names; it reads that one key of
Talon's policy and nothing else of Talon's configuration. A host whose Talon
reads its policy from another file names the dispatcher in ``[idle_wake]
talon_dispatcher``. Named in both places, they must agree. A blank name, or
two that disagree, is a configuration error and the wake does not fire. The dispatcher's
registry holds every run its Talon started, whoever ordered it, so "no job
in flight" means the host's Talon lane is free, and a job dispatched there
for anyone ends the stretch. The work order itself is covered by the wait
ledger when the orchestrator arms an ``a2a:`` watch on the order it sends,
as it must until a sender is woken by its peer's completion (#3755): an
order still awaiting its report is then an armed watch.

The dispatcher is found through the host's agent manager, so it must be an
agent of the same host; one that is not running here, or loads no Talon, is
unknown and the wake does not fire. Its registry is only read: nothing is
written to the dispatcher. With no dispatcher named, or one naming the agent
itself, the agent's own registry is its lane, as on a host with no
dispatcher policy, where each agent runs its own Talon work.

One wake per idle stretch
-------------------------
After the wake is delivered the stretch is spent. It does not fire again
until something starts new work or a person or peer turns up:

* a Talon job is dispatched in the agent's Talon lane (its job id appears,
  whether it is still running or already finished);
* a watch is armed (seen armed at an evaluation), or a wake is delivered
  (a delivered wake means a watch was armed or a job dispatched since);
* a restart request is filed, including one the agent files itself;
* an operator turn or an A2A turn opens.

Each of those starts a new stretch, and the idle clock starts again when the
agent next looks idle. Only new evidence counts: a job id or restart request
that was not in the previous reading, or a wake delivered later than the
newest one read. Evidence that disappears (a pruned job, a deleted row) is not
new work, and a source read for the first time is a baseline, not news, as
is a Talon lane that moved to another agent's registry. A
wake still being delivered, or a turn in progress, restarts the idle clock but
does not start a new stretch, so the triage turn's own delivery can never
re-arm it. The marker is persisted (``idle_wake_state``), so a restart neither
forgets a spent stretch nor re-fires it.

The wake is enqueued, not awaited (see :mod:`kestrel_sovereign.signals.delivery`).
A delivered or coalesced wake spends the stretch, and so does a cooperative
Stop of the triage turn, which the dispatcher reports as coalesced. A failed
one is retried on later ticks, at most :data:`MAX_WAKE_ATTEMPTS` times per
stretch, each attempt under its own durable source event id and dedupe key.
A dispatch that ends in execution-custody evidence (an unknown commit
outcome, revoked authority), including a cancellation caused by it, is
neither: whether its turn ran is decided by the reply it stored.

The evaluation runs inside a cron occurrence, but the wake outlives it. Its
enqueue, its dispatch and turn, and its settlement run as the agent runtime's
work, handed over by the evaluation, and so does the lease heartbeat. On
PostgreSQL a scheduler occurrence's execution admission retires when the
occurrence returns, and work started as its child would be refused from then
on.

One process runs the detector
-----------------------------
The detector sees only its own process: whether a turn is running is
process-local state, and so is a wake's dispatch. A second process could not
see the first one's turns, and could declare the agent idle while an operator
turn runs there. So one process at a time runs the detector, the one holding
its lease (``owner`` in ``idle_wake_state``). The first evaluation takes the
lease and starts a heartbeat that renews it every
:data:`OWNER_HEARTBEAT_INTERVAL` for as long as the process lives. An
evaluation on any other process finds the lease live and does nothing at all:
it observes nothing, writes nothing and dispatches nothing (verdict
``not_owner``). A lease not renewed for :data:`OWNER_STALE_AFTER` is taken
over by the next evaluation that finds it so. A restarted process is a new
owner, so after a restart the detector resumes once the old lease lapses.

A lapsed lease normally means its process stopped, but it can also belong to
a live process whose renewals failed (its database unreachable, its event
loop stalled). The owner guards itself against that: an evaluation that finds
its own lease lapsed renews it before going on, and the owner fires only when
it confirmed its lease within :data:`LEASE_CONFIRMED_WITHIN` (verdict
``lease_unconfirmed`` otherwise). What it cannot guard is a wake it already
enqueued: if its renewals keep failing past the staleness while that triage
turn runs, another process takes the lease over and may retry the wake.

A wake belongs to the process that enqueued it, which settles it as soon as
its dispatch ends. A wake still pending when its owner's lease is taken over
is presumed to have gone down with that process: the new owner counts it as
delivered when its turn stored a reply, retries it when the lookup finds none,
and leaves it pending while the lookup cannot be made.

So two processes that overlap briefly, as in a rolling restart, never both
evaluate or both dispatch. The owner still cannot see a turn running in the
other process; it sees only that turn's opening row, and only when an operator
or a peer opened it. A deployment that runs one agent on more than one replica
at once must keep ``[idle_wake]`` off (the default): the owner does not
observe turns running on the other replicas, and the cron occurrences that
land on them are skipped, so the owner's own evaluations may come too far
apart to see an idle stretch through.

Every write to ``idle_wake_state`` is a compare-and-set on the row's
revision. A takeover also requires the holder and heartbeat it read, and
advances the revision. So two processes taking a lapsed lease at once cannot
both win, a takeover racing the owner's renewal loses to it, the old owner
cannot write from a reading older than the takeover, and an evaluation
working from an older reading cannot undo a settled wake. The loser drops its
write; the next evaluation reads afresh. A renewal moves only the heartbeat,
so it never invalidates the owner's own evaluation in progress.

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

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, owned_task_error
from kestrel_sovereign.config import load_section
from kestrel_sovereign.execution_custody import is_execution_control_error

logger = logging.getLogger(__name__)

#: The kestrel.toml section the detector reads.
CONFIG_SECTION = "idle_wake"
#: The kestrel.toml section holding Talon's operator policy, whose
#: ``dispatcher`` names the host's only Talon dispatcher (kestrel-feature-talon
#: 0.2.14). The detector reads only that one key, to know whose registry is
#: this agent's Talon lane.
TALON_CONFIG_SECTION = "talon"

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

#: How often the process running the detector renews its lease.
OWNER_HEARTBEAT_INTERVAL = timedelta(seconds=30)
#: A lease not renewed for this long belongs to a process that stopped. It is
#: the dispatcher's runtime-owner staleness, and like it compares the owner's
#: clock with the reader's: processes' clocks must agree to well within it.
OWNER_STALE_AFTER = timedelta(minutes=2)
#: An owner fires only when it confirmed its lease this recently. Half the
#: staleness leaves the claim and the enqueue well inside the lease, so no
#: other process can take it over between them.
LEASE_CONFIRMED_WITHIN = OWNER_STALE_AFTER / 2
#: The name of the task renewing the lease. It runs for the life of the
#: process, so the restart coordinator's idle gate counts it as bookkeeping,
#: not work (``_INFRA_TASK_PREFIXES``).
OWNER_HEARTBEAT_TASK_PREFIX = "idle_wake_owner_heartbeat:"
#: The name of the task that owns one wake: it hands the wake to the
#: dispatcher and settles it when its dispatch ends. Like the heartbeat it
#: belongs to the agent's runtime, not to the cron occurrence that fired it.
TRIAGE_TASK_PREFIX = "idle_wake_triage:"
#: Times an owner tries to settle its finished wake when another write to the
#: row came first. If every try loses, the owner's next evaluation settles it.
SETTLE_TRIES = 3

#: A pending wake whose owner stopped before settling it, but whose turn
#: stored its reply: it was delivered.
REPLIED_BEFORE_RESTART = "replied_before_restart"
#: Dispatcher statuses that spend the stretch. ``coalesced`` means an
#: equivalent wake for this exact attempt was already accepted, or the triage
#: turn was stopped cooperatively.
_SPENT_STATUSES = frozenset({"ok", "coalesced", REPLIED_BEFORE_RESTART})
#: Statuses no retry can change.
_HARD_FAIL_STATUSES = frozenset({"dropped_validation", "dropped_cycle"})
#: A pending wake whose owner stopped and whose turn stored no reply: it never
#: finished, so it is retried.
LOST_AT_RESTART = "lost_at_restart"

#: Restart request states in which the agent is waiting on its restart.
RESTART_WORK_STATUSES = frozenset({"pending", "approved", "updating", "executing"})

#: Wake sources whose turn was opened by a peer agent.
A2A_SOURCE_PREFIX = "a2a."

#: How a report names the agent's own Talon registry as the lane it read.
OWN_TALON_REGISTRY = "self"

#: Conversation rows read per evaluation to find newly opened turns.
CONVERSATION_SCAN_ROWS = 200
#: Delivered wait-ledger wakes searched for the most recent chain session.
CHAIN_SESSION_SCAN_ROWS = 20


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdleWakeConfig:
    """``[idle_wake]`` as it applies to one agent.

    ``talon_dispatcher`` is the ``agent_name`` of the agent that runs this
    agent's Talon work, the agent ``[talon.policy] dispatcher`` names; the
    detector reads that agent's Talon registry (see :meth:`IdleWakeDetector.
    _talon_lane`). ``None`` means the agent runs its own. ``config_error``
    says why the configuration could not be read as written; the detector
    then never reads the agent as idle.
    """

    enabled: bool = False
    idle_minutes: int = DEFAULT_IDLE_MINUTES
    talon_dispatcher: Optional[str] = None
    config_error: Optional[str] = None


def _agent_names(agent: Any) -> set:
    names = set()
    for attr in ("agent_name", "_agent_name", "did"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
    return names


def load_idle_wake_config(
    agent: Any,
    section: Optional[Dict[str, Any]] = None,
    talon_section: Optional[Dict[str, Any]] = None,
) -> IdleWakeConfig:
    """Read ``[idle_wake]`` for ``agent``.

    Off unless ``enabled = true``. When ``agents`` lists names (or DIDs), only
    those agents are enabled; an absent or empty list enables every agent this
    host runs, as ``[preturn_state]`` does. ``idle_minutes`` defaults to
    :data:`DEFAULT_IDLE_MINUTES`; a value that is not an integer of at least
    :data:`MIN_IDLE_MINUTES` is refused with a warning and the default used.

    ``talon_dispatcher`` names the agent whose Talon registry holds this
    agent's runs. Unset, it is the agent ``[talon.policy] dispatcher`` names
    in the same kestrel.toml (``talon_section``, read from the file when not
    passed), the file a fleet host's Talon ``config_path`` names. A
    host whose Talon reads its policy from another file sets
    ``talon_dispatcher`` itself. Set in both places, they must name the same
    agent. A blank or non-string value in either, or two that disagree, is a
    configuration error, not "absent": reading the agent's own empty
    registry in the dispatcher's place would report Talon idle while the
    dispatcher's lane runs.
    """
    cfg = load_section(CONFIG_SECTION) if section is None else section
    if talon_section is None:
        talon_section = load_section(TALON_CONFIG_SECTION)
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
    dispatcher, config_error = _talon_dispatcher(cfg, talon_section)
    if config_error:
        logger.warning(
            "[idle_wake] %s; the detector will not read the agent as idle "
            "until its Talon dispatcher is named consistently",
            config_error,
        )
    return IdleWakeConfig(
        enabled=True,
        idle_minutes=raw,
        talon_dispatcher=dispatcher,
        config_error=config_error,
    )


def _agent_name_value(value: Any) -> Tuple[Optional[str], bool]:
    """``(name, valid)`` for an agent-name setting: absent is ``(None, True)``."""
    if value is None:
        return None, True
    if isinstance(value, str) and value.strip():
        return value.strip(), True
    return None, False


def _talon_dispatcher(
    cfg: Dict[str, Any], talon_section: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """The agent running this agent's Talon work, or why it is not known.

    ``[idle_wake] talon_dispatcher`` and ``[talon.policy] dispatcher`` (see
    :func:`load_idle_wake_config`). Returns ``(dispatcher, config_error)``.
    """
    explicit_raw = cfg.get("talon_dispatcher")
    explicit, valid = _agent_name_value(explicit_raw)
    if not valid:
        return None, f"talon_dispatcher = {explicit_raw!r} is not an agent name"
    talon = talon_section if isinstance(talon_section, dict) else {}
    policy_table = talon.get("policy")
    if policy_table is not None and not isinstance(policy_table, dict):
        return None, f"[talon] policy = {policy_table!r} is not a table"
    policy_raw = (policy_table or {}).get("dispatcher")
    policy, valid = _agent_name_value(policy_raw)
    if not valid:
        return None, (
            f"[talon.policy] dispatcher = {policy_raw!r} is not an agent name"
        )
    if explicit is None:
        return policy, None
    if policy is not None and explicit.casefold() != policy.casefold():
        return None, (
            f"talon_dispatcher = {explicit!r}, but [talon.policy] dispatcher "
            f"= {policy!r}"
        )
    return explicit, None


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

    ``owner`` is the process running the detector and ``owner_heartbeat_at``
    when it last renewed its lease. ``pending_signal_id`` is the wake that
    owner enqueued and has not settled yet. ``revision`` is the row's revision
    as read (``0``: no row yet); every write is conditional on it.
    """

    idle_since: Optional[str] = None
    last_evaluated_at: Optional[str] = None
    fired_at: Optional[str] = None
    fired_status: Optional[str] = None
    attempts: int = 0
    pending_signal_id: Optional[str] = None
    owner: Optional[str] = None
    owner_heartbeat_at: Optional[str] = None
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
                owner                TEXT,
                owner_heartbeat_at   TEXT,
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
                   attempts, pending_signal_id, owner,
                   owner_heartbeat_at, evidence, conversation_cursor,
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
            owner=row[6],
            owner_heartbeat_at=row[7],
            evidence=evidence,
            conversation_cursor=int(row[9]) if row[9] is not None else None,
            last_wake_status=row[10],
            last_wake_error=row[11],
            revision=int(row[12] or 0),
        )

    async def save(self, state: IdleWakeState) -> Optional[IdleWakeState]:
        """Write ``state`` if the row is still at the revision it was read at.

        Returns the state as written, its revision advanced, or ``None`` when
        another write got there first: the owner settling its wake, or
        another process taking a lapsed lease. That write stands; the caller
        drops its own. The lease is not written here; only
        :meth:`take_lease` and :meth:`renew` write it, so a save can neither
        hand the lease on nor wind its heartbeat back.
        """
        if state.revision < 1:
            raise ValueError("idle_wake_state: take_lease() creates the row")
        await self._ensure_table()
        written = await self._db.execute(
            """
            UPDATE idle_wake_state SET
                idle_since = ?, last_evaluated_at = ?, fired_at = ?,
                fired_status = ?, attempts = ?, pending_signal_id = ?,
                evidence = ?, conversation_cursor = ?, last_wake_status = ?,
                last_wake_error = ?, updated_at = ?,
                revision = revision + 1
            WHERE agent_id = ? AND revision = ?
            """,
            (
                state.idle_since,
                state.last_evaluated_at,
                state.fired_at,
                state.fired_status,
                int(state.attempts),
                state.pending_signal_id,
                json.dumps(state.evidence, sort_keys=True),
                state.conversation_cursor,
                state.last_wake_status,
                state.last_wake_error,
                _iso(datetime.now(timezone.utc)),
                self._agent_id,
                state.revision,
            ),
        )
        if written != 1:
            return None
        return replace(state, revision=state.revision + 1)

    async def take_lease(
        self, state: IdleWakeState, owner: str, heartbeat_at: str
    ) -> Optional[IdleWakeState]:
        """Make ``owner`` the detector's owner, if the lease is still exactly
        as ``state`` read it.

        That is the same holder and the same heartbeat: a renewal since the
        read means the holder is alive, and the takeover loses to it. Taking
        the lease advances the revision, so the previous owner's writes from
        older readings fail. ``None`` when the takeover lost.
        """
        await self._ensure_table()
        now = _iso(datetime.now(timezone.utc))
        if state.revision == 0:
            written = await self._db.execute(
                """
                INSERT INTO idle_wake_state
                    (agent_id, owner, owner_heartbeat_at, updated_at, revision)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(agent_id) DO NOTHING
                """,
                (self._agent_id, owner, heartbeat_at, now),
            )
        else:
            held_by = "owner IS NULL" if state.owner is None else "owner = ?"
            renewed_at = (
                "owner_heartbeat_at IS NULL"
                if state.owner_heartbeat_at is None
                else "owner_heartbeat_at = ?"
            )
            written = await self._db.execute(
                f"""
                UPDATE idle_wake_state SET
                    owner = ?, owner_heartbeat_at = ?, updated_at = ?,
                    revision = revision + 1
                WHERE agent_id = ? AND revision = ? AND {held_by}
                      AND {renewed_at}
                """,
                (owner, heartbeat_at, now, self._agent_id, state.revision)
                + tuple(
                    value
                    for value in (state.owner, state.owner_heartbeat_at)
                    if value is not None
                ),
            )
        if written != 1:
            return None
        return replace(
            state,
            owner=owner,
            owner_heartbeat_at=heartbeat_at,
            revision=state.revision + 1,
        )

    async def renew(self, owner: str, heartbeat_at: str) -> bool:
        """Renew ``owner``'s lease on the detector.

        ``False`` when ``owner`` no longer holds it: another process took it
        over after it lapsed. A renewal moves only the heartbeat, not the
        revision, so it never invalidates the owner's own evaluation in
        progress; a takeover racing it is refused because the heartbeat it
        read is gone.
        """
        await self._ensure_table()
        written = await self._db.execute(
            """
            UPDATE idle_wake_state SET owner_heartbeat_at = ?
            WHERE agent_id = ? AND owner = ?
            """,
            (heartbeat_at, self._agent_id, owner),
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
    #: Whose Talon registry was read: :data:`OWN_TALON_REGISTRY`, the
    #: dispatcher's name, or ``None`` when none was.
    talon_registry: Optional[str] = None

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


def _refusal_outcome(
    error: Optional[BaseException],
) -> Tuple[Optional[str], Optional[str]]:
    """How a wake that never reached the dispatcher ended: ``error`` is what
    stopped its hand-over. Nothing started, so any of these is retried.

    ``None`` for execution-custody evidence, as for a dispatch (see
    :func:`_dispatch_outcome`): the reply lookup then finds no reply and the
    wake is retried as lost.
    """
    if error is None:
        return "dispatcher_raised", "the wake was never handed to the dispatcher"
    if is_execution_control_error(error):
        return None, f"dispatch_outcome_unknown: {_describe(error)}"
    if isinstance(error, asyncio.CancelledError):
        return "dispatch_cancelled", None
    return "dispatcher_raised", _describe(error)


def _dispatch_outcome(task: asyncio.Future) -> Tuple[Optional[str], Optional[str]]:
    """The settled status of a finished wake dispatch, and its error.

    ``None`` when the dispatch ended in execution-custody evidence (an
    unknown commit outcome, revoked authority): whether its turn ran is not
    known from the dispatch, so it is not settled as a failure to retry. Its
    turn's stored reply decides it (:meth:`IdleWakeDetector._reply_outcome`).

    The evidence is read through :func:`owned_task_error`, never
    ``task.result()``. A dispatch cancelled with custody evidence as its
    cause is that evidence, not a cancellation to retry, and asyncio hands a
    cancelled task's original exception only to the first reader: the
    agent's own completion callback reads it before this does.
    """
    error = owned_task_error(task)
    if error is None:
        result = task.result()
    elif is_execution_control_error(error):
        return None, f"dispatch_outcome_unknown: {_describe(error)}"
    elif isinstance(error, asyncio.CancelledError):
        return "dispatch_cancelled", None
    else:
        return "dispatcher_raised", _describe(error)
    status = getattr(getattr(result, "status", None), "value", None)
    if not isinstance(status, str):
        return "tooling_error", f"dispatch returned no usable status: {result!r}"
    return status, getattr(result, "error", None) or None


@dataclass(eq=False)
class _OwnedWake:
    """A wake this process fired, held by its triage task.

    ``task`` is the triage task (:meth:`IdleWakeDetector._run_wake`).
    ``handed_over`` resolves once the dispatcher accepted the wake or refused
    it. ``dispatch`` is the dispatch task once accepted, even when
    ``enqueue_signal`` raised after accepting it (``enqueue_error``).
    ``enqueue_error_seen`` says the evaluation that fired the wake read that
    error, so the triage task need not raise it again.
    """

    handed_over: asyncio.Future
    task: Optional[asyncio.Task] = None
    dispatch: Optional[asyncio.Future] = None
    enqueue_error: Optional[BaseException] = None
    enqueue_error_seen: bool = False

    def done(self) -> bool:
        """Whether nothing of this wake is running any more."""
        return (self.task is None or self.task.done()) and (
            self.dispatch is None or self.dispatch.done()
        )

    def outcome(self) -> Tuple[Optional[str], Optional[str]]:
        """How the finished wake ended (see :func:`_dispatch_outcome`)."""
        if self.dispatch is not None:
            return _dispatch_outcome(self.dispatch)
        error = self.enqueue_error
        if error is None and self.task is not None and self.task.done():
            # Stopped before it reached the dispatcher: cancelled, or refused
            # by the runtime it was handed to.
            error = owned_task_error(self.task)
        return _refusal_outcome(error)


class IdleWakeDetector:
    """One agent's idle detector, held on ``agent._idle_wake_detector``.

    It evaluates only while the database says this process holds the
    detector's lease, which a heartbeat task renews for the life of the
    process, and it fires only while it has recently confirmed that lease. It
    owns the wakes it fires: each has a triage task that hands it to the
    dispatcher and settles it when its dispatch ends, and the detector keeps
    each wake's in-memory record until it is settled.

    The heartbeat and the triage tasks outlive the cron occurrence whose
    evaluation started them, so they are handed to the agent's runtime
    (:meth:`_track_resident`) rather than started as children of the
    occurrence.
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
        # This process, as the holder of the detector's lease. A restarted
        # process is a new owner: it cannot renew the old one's lease.
        self._owner = f"idle-wake:{secrets.token_urlsafe(12)}"
        self._heartbeat_seconds = heartbeat_interval.total_seconds()
        self._heartbeat: Optional[asyncio.Task] = None
        # When this process last confirmed its lease (took or renewed it),
        # by its own clock.
        self._confirmed_at: Optional[datetime] = None
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
            self.stand_down()
            return {"skipped": "disabled", "fired": False}
        async with self._lock:
            return await self._tick(config, pending_followups)

    def stand_down(self) -> None:
        """Stop renewing the lease: the detector was disabled. It lapses
        after :data:`OWNER_STALE_AFTER`; wakes already enqueued still settle.
        Nothing is written, so a disabled detector touches no state."""
        if self._heartbeat is not None and not self._heartbeat.done():
            self._heartbeat.cancel()
        self._heartbeat = None

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

        # --- The lease: one process runs the detector --------------------
        if state.owner != self._owner:
            if state.owner is not None and not self._lease_stale(state, now):
                # Another live process runs the detector. This one cannot see
                # that one's turns, so it does not evaluate at all.
                report.update(
                    verdict="not_owner",
                    owner_heartbeat_at=state.owner_heartbeat_at,
                )
                return report
            previous, renewed = state.owner, state.owner_heartbeat_at
            acquired = await self._state_store.take_lease(
                state, self._owner, _iso(now)
            )
            if acquired is None:
                report.update(verdict="state_conflict", superseded="take_lease")
                return report
            state = acquired
            self._confirm(now)
            if previous is not None:
                logger.info(
                    "idle wake: took over the detector; %s last renewed its "
                    "lease at %s", previous, renewed,
                )
                report["took_over"] = True
        elif self._lease_stale(state, now):
            # This process's own renewals have not landed for a while, so
            # another may be about to take the lease over. Renew it before
            # going on; a renewal that finds it taken stops here.
            if not await self._renew():
                report.update(verdict="state_conflict", superseded="renew")
                return report
            state = replace(state, owner_heartbeat_at=_iso(now))
            report["renewed_lapsed_lease"] = True
        self._start_heartbeat()

        # --- A wake enqueued earlier and not settled yet ----------------
        if state.pending_signal_id:
            signal_id = state.pending_signal_id
            wake = self._pending.get(signal_id)
            if wake is not None and not wake.done():
                # The triage turn is running here: the agent is busy, and
                # the stretch's own wake must not start a new one.
                return await self._wake_in_flight(state, now, report)
            if wake is not None:
                # Settling it when the dispatch ended failed; settle it now.
                status, error = wake.outcome()
                if status is None:
                    # Its outcome is unknown: its turn's reply decides it.
                    status, error = await self._reply_outcome(signal_id)
            else:
                # Enqueued by an earlier owner, whose lease lapsed: the
                # wake's dispatch stopped with that process.
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

        observation = await self._observe(state, config, pending_followups)
        report.update(
            work=observation.work,
            deliveries=observation.deliveries,
            unknown=observation.unknown,
            activity=observation.activity,
            talon_registry=observation.talon_registry,
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
        write landed since the row was read (a process taking over a lapsed
        lease), and it stands. The next evaluation reads afresh.
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
    ) -> Dict[str, Any]:
        """The agent is busy with the stretch's own wake. The idle clock
        restarts; the stretch, and its attempt count, go on."""
        return await self._commit(
            replace(state, idle_since=None, last_evaluated_at=_iso(now)),
            report,
            "wake_in_flight",
            signal_id=state.pending_signal_id,
        )

    @staticmethod
    def _lease_stale(state: IdleWakeState, now: datetime) -> bool:
        """Whether the detector's owner has stopped renewing its lease."""
        heartbeat = _parse(state.owner_heartbeat_at)
        return heartbeat is None or now - heartbeat > OWNER_STALE_AFTER

    def _confirm(self, at: datetime) -> None:
        """Record that this process held its lease at ``at``."""
        if self._confirmed_at is None or at > self._confirmed_at:
            self._confirmed_at = at

    async def _renew(self) -> bool:
        """Renew this process's lease now. ``False``: another process holds
        it."""
        at = self._clock()
        held = await self._state_store.renew(self._owner, _iso(at))
        if held:
            self._confirm(at)
        return held

    def _start_heartbeat(self) -> None:
        """Keep this process's lease renewed, unless that is already so."""
        if self._heartbeat is not None and not self._heartbeat.done():
            return
        self._heartbeat = self._track_resident(
            self._hold_lease(),
            name=f"{OWNER_HEARTBEAT_TASK_PREFIX}{self._owner}",
        )

    def _track_resident(self, coro: Any, *, name: str) -> asyncio.Task:
        """Start ``coro`` as the agent runtime's work, not the occurrence's.

        Evaluations run inside a cron occurrence, and what they start (the
        heartbeat, a wake's triage task, and through it the wake's dispatch
        and turn) runs on after the occurrence returns. On PostgreSQL a
        scheduler occurrence carries its own execution admission, which
        retires as it returns, and a task started as its child carries that
        admission too: every write it made afterwards would be refused.
        ``_track_runtime_task`` hands the task over instead. It refuses
        unless every admission held now is live, and then runs the task under
        the agent's own runtime admission alone, once the agent is ready. An
        agent without that hand-over (a host predating #3570, a test double)
        runs it as ordinary background work.
        """
        if callable(getattr(type(self._agent), "_track_runtime_task", None)):
            return self._agent._track_runtime_task(coro, name=name)
        return self._agent._track_background_task(coro, name=name)

    async def _hold_lease(self) -> None:
        """Renew the lease every heartbeat interval while the process lives.

        It does not wait for the evaluation lock, so a slow evaluation (a
        large Talon registry, a contended database) cannot hold renewals back
        until the lease lapses. A renewal that fails is retried at the next
        interval rather than ending the heartbeat: a lease left to lapse while
        this process lives would let another one start evaluating, blind to
        the turns running here. A renewal that finds the lease taken ends it:
        this process evaluates nothing more unless that lease lapses in turn.

        Execution-custody evidence is not a failed renewal: a renewal
        cancelled while it commits ends in an unknown commit outcome, not in
        ``CancelledError``. That, and any failure while the task is being
        cancelled, ends the heartbeat, as does a renewal that returns while
        the task is being cancelled; going round again would leave a shutdown
        waiting on a loop that swallowed its cancellation.
        """
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                held = await self._renew()
            except Exception as exc:  # noqa: BLE001 - retried next interval
                task = asyncio.current_task()
                if is_execution_control_error(exc) or (
                    task is not None and task.cancelling()
                ):
                    raise
                logger.warning(
                    "idle wake: could not renew the detector's lease: %s",
                    _describe(exc),
                )
                continue
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                # The renewal absorbed this task's cancellation and returned.
                raise asyncio.CancelledError
            if not held:
                logger.warning(
                    "idle wake: another process took over the detector after "
                    "this one's lease lapsed; this process stops evaluating "
                    "while that one's lease is live"
                )
                return

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
        confirmed = self._confirmed_at
        unconfirmed_for = (
            self._clock() - confirmed if confirmed is not None else None
        )
        if unconfirmed_for is None or unconfirmed_for > LEASE_CONFIRMED_WITHIN:
            # Renewals are failing: another process may take the lease over
            # at any moment, and would read this wake as lost. Fire once the
            # lease is confirmed again.
            return await self._commit(
                state,
                report,
                "lease_unconfirmed",
                lease_confirmed_at=_iso(confirmed) if confirmed else None,
            )
        # The claim, durable before the enqueue: a crash between the two is a
        # lost wake that is recovered and retried, never an unrecorded one.
        # Its compare-and-set fails if the lease was taken since it was read.
        claimed = await self._state_store.save(
            replace(state, attempts=attempt, pending_signal_id=signal.id)
        )
        if claimed is None:
            report.update(verdict="state_conflict", superseded="fired")
            return report
        state = claimed
        wake = _OwnedWake(handed_over=asyncio.get_running_loop().create_future())
        try:
            wake.task = self._track_resident(
                self._run_wake(wake, dispatcher, signal, f"idle:{period}#{attempt}"),
                name=f"{TRIAGE_TASK_PREFIX}{signal.id}",
            )
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            # The runtime refused the hand-over: nothing started.
            if is_execution_control_error(exc):
                # The claim stays recorded; the next evaluation finds no
                # reply and retries it.
                raise
            return await self._refused(state, report, exc, now)
        self._pending[signal.id] = wake
        # Wait for the hand-over, not for the turn. Should this evaluation
        # be cancelled meanwhile, the triage task still owns the wake: it is
        # recorded, so the next evaluation finds it running, not lost.
        await asyncio.wait(
            {wake.handed_over, wake.task}, return_when=asyncio.FIRST_COMPLETED
        )
        error = wake.enqueue_error
        wake.enqueue_error_seen = True
        if wake.dispatch is None:
            # Nothing started: refused by the dispatcher, or the triage task
            # stopped before it reached it.
            self._pending.pop(signal.id, None)
            if error is None and wake.task.done():
                error = owned_task_error(wake.task)
            if error is not None and is_execution_control_error(error):
                # As above: the claim stays recorded and is retried.
                raise error
            return await self._refused(state, report, error, now)
        if error is not None:
            # The agent accepted the dispatch, then failed to hand it back:
            # it runs, so it is this process's wake, settled when it ends.
            # Retrying it now could start a second triage turn.
            logger.warning(
                "idle wake: enqueue_signal raised after accepting %s; "
                "settling it from its dispatch: %s", signal.id, _describe(error),
            )
            report["wake_error"] = _describe(error)
            if is_execution_control_error(error):
                raise error
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

    async def _refused(
        self,
        state: IdleWakeState,
        report: Dict[str, Any],
        error: Optional[BaseException],
        now: datetime,
    ) -> Dict[str, Any]:
        """Settle a wake that never reached the dispatcher, as one attempt."""
        status, detail = _refusal_outcome(error)
        logger.warning("idle wake: the wake was not handed over: %s", detail or status)
        settled = self._settle(state, status, detail, now)
        report.update(wake_error=detail or status, fired_at=settled.fired_at)
        return await self._commit(settled, report, "dispatch_error")

    async def _run_wake(
        self,
        wake: _OwnedWake,
        dispatcher: Any,
        signal: Any,
        source_event_id: str,
    ) -> None:
        """The triage task: own one wake from the agent's runtime.

        It hands the wake to the dispatcher here, so the dispatch, and the
        triage turn it runs, are the runtime's work as well
        (:meth:`_track_resident`). Then it settles the wake as soon as the
        dispatch ends, so the outcome is in the row before the next
        evaluation reads it.

        What ``enqueue_signal`` raised is the evaluation's to report, and it
        reads it under the detector's lock. Execution-custody evidence the
        evaluation never read (it was cancelled while waiting) is raised here
        once the wake is settled, so the agent's own harvest keeps it.
        """
        admission = OwnedTaskAdmission()
        try:
            handle = await dispatcher.enqueue_signal(
                signal, source_event_id=source_event_id, admission=admission
            )
        except Exception as exc:  # noqa: BLE001 - the evaluation reports it
            wake.enqueue_error = exc
        else:
            wake.dispatch = handle.task
        finally:
            if wake.dispatch is None and admission.task is not None:
                # Accepted, then not handed back: it runs.
                wake.dispatch = admission.task
            if not wake.handed_over.done():
                wake.handed_over.set_result(None)
        if wake.dispatch is not None:
            await asyncio.wait({wake.dispatch})
        async with self._lock:
            if wake.dispatch is not None:
                await self._settle_own(signal.id, wake)
            unread = None if wake.enqueue_error_seen else wake.enqueue_error
        if unread is not None and is_execution_control_error(unread):
            raise unread

    async def _settle_own(self, signal_id: str, wake: _OwnedWake) -> None:
        """Record the outcome of this process's own finished wake."""
        status, error = wake.outcome()
        if status is None:
            # Its outcome is unknown: the next evaluation decides it from its
            # turn's reply.
            return
        for _ in range(SETTLE_TRIES):
            try:
                state = await self._state_store.load()
                if (
                    state.pending_signal_id != signal_id
                    or state.owner != self._owner
                ):
                    break  # already settled, or the lease was taken over
                settled = self._settle(state, status, error, self._clock())
                if await self._state_store.save(settled) is not None:
                    break
            except Exception as exc:  # noqa: BLE001 - an evaluation settles it
                if is_execution_control_error(exc):
                    raise
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
        """How a wake whose owner stopped before settling it ended.

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
            if is_execution_control_error(exc):
                raise
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
            fired_at=fired_at,
            fired_status=fired_status,
            last_wake_status=status,
            last_wake_error=error,
        )

    # ------------------------------------------------------------------

    async def _observe(
        self,
        state: IdleWakeState,
        config: IdleWakeConfig,
        pending_followups: Optional[Callable[[], Awaitable[int]]],
    ) -> Observation:
        obs = Observation()
        if config.config_error:
            obs.unknown.append(f"idle_wake_misconfigured: {config.config_error}")
        await self._observe_privacy_and_hold(obs)
        await self._observe_conversation(state, obs)
        await self._observe_wait_ledger(obs)
        await self._observe_restarts(obs)
        await self._observe_durable_deliveries(obs)
        # This process's turns only; see "One process runs the detector".
        live_turn = getattr(self._agent, "_live_turn_id", None)
        if isinstance(live_turn, str) and live_turn:
            obs.deliveries.append("turn_in_progress")
        if pending_followups is not None:
            try:
                count = await pending_followups()
            except Exception as exc:  # noqa: BLE001 - fail closed
                if is_execution_control_error(exc):
                    raise
                obs.unknown.append(f"self_followup_unreadable: {_describe(exc)}")
            else:
                if count:
                    obs.deliveries.append(f"self_followup_pending:{count}")
        # The Talon read is the expensive one (it reads every finished job's
        # log). Skip it when the agent is already known not to be idle; its
        # evidence then stays where it was and is compared next time.
        if obs.idle:
            await self._observe_talon(obs, config)
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
            if is_execution_control_error(exc):
                raise
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
            if is_execution_control_error(exc):
                raise
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
            if is_execution_control_error(exc):
                raise
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
            if is_execution_control_error(exc):
                raise
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
                if is_execution_control_error(exc):
                    raise
                obs.unknown.append(
                    f"durable_deliveries_unreadable:{consumer_id}: {_describe(exc)}"
                )
                continue
            if rows:
                obs.deliveries.append(f"durable_delivery_pending:{consumer_id}")

    def _talon_lane(
        self, config: IdleWakeConfig
    ) -> Tuple[Optional[str], Any, Optional[str]]:
        """Whose Talon registry holds this agent's runs, and its provider.

        Returns ``(registry, provider, unknown)``. ``registry`` names whose
        registry it is (:data:`OWN_TALON_REGISTRY`, or the dispatcher's
        name), ``provider`` is that agent's ``talon`` wait provider, and
        ``unknown`` says why the lane cannot be read.

        With no dispatcher named (``config.talon_dispatcher``, taken from
        ``[talon.policy] dispatcher`` when unset), or one naming this agent, the agent
        runs its own Talon work and its own provider is the lane; without one
        it runs no Talon, so there is nothing to read (``(None, None,
        None)``). Otherwise the work runs on the dispatcher (#3758): a
        non-dispatcher's own registry stays empty while the dispatcher's
        lane is busy, so it is never read in the dispatcher's place. The
        dispatcher must be an agent of this host, found through the host's
        agent manager; one that is not loaded here, or runs no Talon, is
        unknown, not idle.
        """
        from kestrel_sovereign.features.strategic_memory.run_history import (
            TALON_WAIT_KIND,
        )

        dispatcher = config.talon_dispatcher
        if dispatcher is None or dispatcher.casefold() in {
            name.casefold() for name in _agent_names(self._agent)
        }:
            registry = getattr(self._agent, "wait_registry", None)
            provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
            if provider is None:
                return None, None, None
            return OWN_TALON_REGISTRY, provider, None
        manager = getattr(self._agent, "_agent_manager", None)
        get_agent = getattr(manager, "get_agent", None)
        if not callable(get_agent):
            return dispatcher, None, (
                f"talon_dispatcher_unreachable: {dispatcher} is not an agent "
                f"of this host"
            )
        try:
            hosted = get_agent(dispatcher)
        except Exception as exc:  # noqa: BLE001 - fail closed
            if is_execution_control_error(exc):
                raise
            return dispatcher, None, (
                f"talon_dispatcher_unreachable: {dispatcher}: {_describe(exc)}"
            )
        if hosted is None:
            return dispatcher, None, (
                f"talon_dispatcher_unreachable: {dispatcher} is not running on "
                f"this host"
            )
        registry = getattr(hosted, "wait_registry", None)
        provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
        if provider is None:
            return dispatcher, None, (
                f"talon_dispatcher_has_no_talon: {dispatcher} has no "
                f"{TALON_WAIT_KIND!r} wait provider"
            )
        return dispatcher, provider, None

    async def _observe_talon(self, obs: Observation, config: IdleWakeConfig) -> None:
        """Whether a Talon job is running in this agent's Talon lane.

        Read once per evaluation through the lane provider's read-only
        ``finished_runs()``, never ``poll()`` (it reaps and pushes) or
        ``active_handles()`` (see :data:`~kestrel_sovereign.features.
        strategic_memory.run_history.UNFINISHED_JOBS_KEY`). Nothing is
        written to the lane's agent.
        """
        from kestrel_sovereign.features.strategic_memory.run_history import (
            FINISHED_RUNS_METHOD,
            UNFINISHED_JOBS_KEY,
            UNFINISHED_JOBS_REQUIREMENT,
        )

        lane, provider, unknown = self._talon_lane(config)
        obs.talon_registry = lane
        if unknown is not None:
            obs.unknown.append(unknown)
            return
        if provider is None:
            # No Talon on this agent: no jobs, nothing read.
            return
        read = getattr(provider, FINISHED_RUNS_METHOD, None)
        if not callable(read):
            obs.unknown.append(
                f"talon_provider_outdated: no {FINISHED_RUNS_METHOD}() "
                f"(needs {UNFINISHED_JOBS_REQUIREMENT})"
            )
            return
        try:
            report = await read()
        except Exception as exc:  # noqa: BLE001 - fail closed
            if is_execution_control_error(exc):
                raise
            obs.unknown.append(f"talon_registry_unreadable: {_describe(exc)}")
            return
        if not isinstance(report, dict) or report.get("complete") is not True:
            reason = report.get("reason") if isinstance(report, dict) else None
            obs.unknown.append(f"talon_registry_unreadable: {reason or 'incomplete'}")
            return
        if UNFINISHED_JOBS_KEY not in report:
            # The running jobs must come from the read ``complete`` vouches
            # for. ``active_handles()`` reads the registry again and cannot
            # say that read failed (see run_history.UNFINISHED_JOBS_KEY).
            obs.unknown.append(
                f"talon_provider_outdated: {FINISHED_RUNS_METHOD}() lists no "
                f"{UNFINISHED_JOBS_KEY!r} jobs (needs {UNFINISHED_JOBS_REQUIREMENT})"
            )
            return
        runs = report.get("runs")
        unfinished = report.get(UNFINISHED_JOBS_KEY)
        if not isinstance(runs, list) or not isinstance(unfinished, list):
            obs.unknown.append("talon_registry_unreadable: a list is missing")
            return
        # Every entry is a record naming its job: ``{"job_id", ...}``. An entry
        # that names none could hide a job still running.
        jobs = [
            entry.get("job_id") if isinstance(entry, dict) else None
            for entry in [*runs, *unfinished]
        ]
        if not all(isinstance(job, str) and job.strip() for job in jobs):
            obs.unknown.append("talon_registry_unreadable: an entry names no job")
            return
        running = sorted({job for job in jobs[len(runs):]})
        obs.work.extend(f"talon_job:{job}" for job in running)
        obs.evidence["talon"] = _short_ids(jobs)
        obs.evidence["talon_registry"] = lane

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
            if source == "talon" and "talon" in obs.evidence and (
                obs.evidence.get("talon_registry")
                != state.evidence.get("talon_registry")
            ):
                # The lane moved to another agent's registry (or this is the
                # first read naming one): its jobs are a baseline, not news.
                continue
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
            if is_execution_control_error(exc):
                raise
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
    ``pending_followups`` counts the agent's ``self_followup`` wakes the
    scheduler has not finished; the scheduler owns that table, so it supplies
    the count.
    """
    if config is None:
        config = load_idle_wake_config(agent)
    detector = getattr(agent, "_idle_wake_detector", None)
    if not config.enabled:
        if detector is not None:
            detector.stand_down()
        return {"skipped": "disabled", "fired": False}
    if detector is None:
        detector = IdleWakeDetector(agent)
        agent._idle_wake_detector = detector
    return await detector.tick(config, pending_followups=pending_followups)
