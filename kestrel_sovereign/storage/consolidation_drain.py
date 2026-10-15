"""Durable evidence that a timed-out consolidation pass left database work running (#2978).

The memory-consolidation deadline cancels its pass, but cancelling an
``aiosqlite`` await does not stop a statement already running on the
connection's worker thread. The backend hands that statement's rollback to a
retained drain and fences the connection behind it, and
``MemorySystem.consolidate`` keeps the MEMORY lock until the drain proves the
statement has returned, or isolates the connection when it cannot wait any
longer.

While that is happening the agent database itself is fenced, so the evidence
cannot be written there. It is a small JSON file beside the SQLite file
instead (``<database>.consolidation-drain.json``), replaced atomically, and
read by the ``memory_system`` health check:

* ``draining`` -- the pass timed out with database work still running, and the
  MEMORY lock is held while it finishes;
* ``isolated`` -- the lock was released before the work finished, so the
  connection fails every operation closed until it does (``isolation`` says
  why: the cleanup deadline ran out, the caller cancelled the wait, or the
  wait itself failed);
* ``done`` -- the work has settled; ``cleanup`` says how
  (``finished``, ``failed``, ``retired`` when the connection was closed under
  it, or ``owner_retired`` when the record was reconciled, below).

A record names what owns its work: the process (PID, start instant, host and
PID namespace, as a serving record does, plus this process's incarnation id)
and the connection generation the statement runs on. Only that owner moves the
record on, and a process that crashes mid-drain never does. So a record still
``draining`` or ``isolated`` is believed only while its owner may be alive:
:func:`reconcile_consolidation_drain` closes it out as ``done`` /
``owner_retired`` once the owning process has provably exited (its worker
thread, and the statement, went with it) or, in the owning process, once no
pass there still holds it. The original fields stay, so the record remains
evidence of what happened; it just stops claiming that anything is running.
``MemorySystem.initialize`` reconciles at startup, and the health check
reconciles every time it reads.

The record holds no memory content: identifiers, timestamps and states only.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from kestrel_sovereign.audit_time import utc_now_iso
from kestrel_sovereign.multi_agent.liveness import (
    process_identity,
    recorded_process_exited,
)

from .db.sqlite import CANCELLATION_CLEANUP_FINISHED

logger = logging.getLogger(__name__)

#: Appended to the SQLite file's name to name its record.
DRAIN_RECORD_SUFFIX = ".consolidation-drain.json"

DRAIN_STATE_DRAINING = "draining"
DRAIN_STATE_ISOLATED = "isolated"
DRAIN_STATE_DONE = "done"

#: The states that claim database work is still running.
_UNSETTLED_STATES = frozenset({DRAIN_STATE_DRAINING, DRAIN_STATE_ISOLATED})

#: Why a connection was isolated while its abandoned work was still running.
ISOLATION_CLEANUP_DEADLINE = "cleanup_deadline"
ISOLATION_CALLER_CANCELLED = "caller_cancelled"
ISOLATION_WAIT_FAILED = "wait_failed"

#: ``cleanup`` of a record closed out because its owner is gone.
CLEANUP_OWNER_RETIRED = "owner_retired"

_RECORD_VERSION = 1

#: This process's incarnation. Tells its own records from those of an earlier
#: process that ran under the same PID, whatever its start time reads as.
_RUNTIME_ID = uuid.uuid4().hex

#: Passes this process owns whose settlement is not yet durably recorded, by
#: pass id. A pass is entered when it first writes its record and leaves once
#: its ``done`` state is on disk, so its in-memory record outranks a file a
#: failed write left behind.
_OWNED_PASSES: Dict[str, "ConsolidationDrain"] = {}


def consolidation_drain_record_path(db: Any) -> Optional[Path]:
    """Where the record for ``db`` lives, or ``None`` when it has no file.

    Only a file-backed SQLite database can leave work running on a worker
    thread after its await is cancelled, and only a file has a directory to
    hold durable evidence.
    """
    backend = getattr(db, "backend", None)
    if getattr(backend, "backend_type", None) != "sqlite":
        return None
    database = getattr(backend, "db_path", None)
    if not isinstance(database, str) or database in ("", ":memory:"):
        return None
    path = Path(database)
    return path.with_name(path.name + DRAIN_RECORD_SUFFIX)


def database_connection_generation(db: Any) -> Optional[str]:
    """The generation of ``db``'s open connection, or ``None`` if unknown."""
    generation = getattr(getattr(db, "backend", None), "connection_generation", None)
    return generation if isinstance(generation, str) else None


def new_consolidation_drain_record(
    *,
    agent_id: str,
    database: Optional[str],
    connection: Optional[str],
    deadline_seconds: float,
) -> Dict[str, Any]:
    """The record of a pass that just timed out with database work running.

    Owned by this process, on the connection generation ``connection``.
    """
    return {
        "version": _RECORD_VERSION,
        "pass_id": uuid.uuid4().hex,
        "agent_id": agent_id,
        "database": database,
        "owner": {**process_identity(), "runtime_id": _RUNTIME_ID},
        "connection": connection,
        "deadline_seconds": deadline_seconds,
        "timed_out_at": utc_now_iso(),
        "state": DRAIN_STATE_DRAINING,
        "isolated_at": None,
        "isolation": None,
        "finished_at": None,
        "cleanup": None,
    }


def isolated_consolidation_drain_record(
    record: Mapping[str, Any], reason: str
) -> Dict[str, Any]:
    """``record`` once its connection is isolated for ``reason``."""
    return {
        **record,
        "state": DRAIN_STATE_ISOLATED,
        "isolated_at": utc_now_iso(),
        "isolation": reason,
    }


def done_consolidation_drain_record(
    record: Mapping[str, Any], cleanup: str
) -> Dict[str, Any]:
    """``record`` once its abandoned work has settled as ``cleanup``."""
    return {
        **record,
        "state": DRAIN_STATE_DONE,
        "finished_at": utc_now_iso(),
        "cleanup": cleanup,
    }


def reconciled_consolidation_drain_record(
    record: Mapping[str, Any], proof: str
) -> Dict[str, Any]:
    """``record`` closed out because ``proof`` shows its owner is gone.

    When the work ended is unknown, so ``finished_at`` stays unset;
    ``reconciled_from`` keeps the state the owner last recorded.
    """
    return {
        **record,
        "state": DRAIN_STATE_DONE,
        "cleanup": CLEANUP_OWNER_RETIRED,
        "reconciled_at": utc_now_iso(),
        "reconciled_from": record.get("state"),
        "reconciliation": proof,
        "reconciled_by": {**process_identity(), "runtime_id": _RUNTIME_ID},
    }


def write_consolidation_drain_record(
    path: Path, record: Mapping[str, Any]
) -> None:
    """Replace the record at ``path`` with ``record``, durably.

    Written aside, fsynced, renamed over the old record and the directory
    fsynced, so a reader sees the previous record or this one, never part of
    either, and a crash cannot lose a state that was reported written.

    Raises:
        OSError: The record could not be written.
    """
    staging = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(staging, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(record), handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def read_consolidation_drain_record(path: Path) -> Optional[Dict[str, Any]]:
    """The record at ``path``, or ``None`` when there is none.

    Raises:
        OSError: The record exists but cannot be read.
        ValueError: The record is not a JSON object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(raw)
    except RecursionError as exc:
        # Python 3.11's decoder raises this, not ValueError, on deep nesting.
        raise ValueError(f"{path} nests too deeply to be a record") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return record


class ConsolidationDrain:
    """The durable record of one timed-out pass's abandoned work.

    Kept in step with the work it describes: ``draining`` when written,
    ``isolated`` if the MEMORY lock is released before the work finishes, and
    ``done`` once it settles, which may happen long after the pass returned.
    A write that fails is reported, never raised: raising would replace the
    deadline outcome the caller must see, and the settlement it records
    happens either way. Until its settlement is on disk the pass stays in
    this process's :data:`_OWNED_PASSES`, so a reader here trusts it over
    the file, and :func:`reconcile_consolidation_drain` retries the write.
    """

    def __init__(
        self, db: Any, *, agent_id: str, deadline_seconds: float
    ) -> None:
        self.path = consolidation_drain_record_path(db)
        self.record = new_consolidation_drain_record(
            agent_id=agent_id,
            database=getattr(getattr(db, "backend", None), "db_path", None),
            connection=database_connection_generation(db),
            deadline_seconds=deadline_seconds,
        )
        self.settled = False

    @property
    def pass_id(self) -> str:
        return self.record["pass_id"]

    def write(self) -> bool:
        """Persist the current state; whether it reached the disk."""
        if self.path is None:
            logger.warning(
                "No durable location for the memory consolidation drain "
                "record (database has no file); state is %s",
                self.record["state"],
            )
            return False
        _OWNED_PASSES[self.pass_id] = self
        try:
            write_consolidation_drain_record(self.path, self.record)
        except OSError:
            logger.error(
                "Could not persist the memory consolidation drain record "
                "%s (state %s)",
                self.path,
                self.record["state"],
                exc_info=True,
            )
            return False
        if self.settled:
            _OWNED_PASSES.pop(self.pass_id, None)
        return True

    def isolate(self, reason: str) -> None:
        """The lock is being released while the work still runs."""
        if self.settled:
            return
        self.record = isolated_consolidation_drain_record(self.record, reason)
        self.write()
        logger.warning(
            "Memory consolidation timed out after %gs and its database work "
            "is still running; the database connection refuses all work "
            "until it finishes, and the memory lock is released (%s; "
            "record: %s)",
            self.record["deadline_seconds"],
            reason,
            self.path,
        )

    def settle(self, cleanup: str) -> None:
        """The work settled as ``cleanup``; runs on the event loop."""
        if self.settled:
            return
        self.settled = True
        was_isolated = self.record["state"] != DRAIN_STATE_DRAINING
        self.record = done_consolidation_drain_record(self.record, cleanup)
        self.write()
        if cleanup != CANCELLATION_CLEANUP_FINISHED:
            logger.error(
                "Database work left running by a timed-out memory "
                "consolidation pass settled as %r; the database connection "
                "refuses work until it is reconnected (record: %s)",
                cleanup,
                self.path,
            )
        elif was_isolated:
            logger.warning(
                "Database work left running by a timed-out memory "
                "consolidation pass has finished; its isolated database "
                "connection accepts work again (record: %s)",
                self.path,
            )
        else:
            logger.info(
                "Database work left running by the timed-out memory "
                "consolidation pass finished; releasing the memory lock",
            )


@dataclass(frozen=True)
class ConsolidationDrainStatus:
    """A drain record, as reconciled against the owner of its work.

    ``unsettled`` is True while that work may still be running. ``holder``
    says who holds it then: ``None`` when it runs on the very connection the
    record was read for (whose state the record describes), otherwise a
    description of the other connection or process.
    """

    path: Path
    record: Dict[str, Any]
    unsettled: bool
    holder: Optional[str]


def reconcile_consolidation_drain(db: Any) -> Optional[ConsolidationDrainStatus]:
    """The drain record for ``db``, with its owner's liveness taken into account.

    ``None`` when ``db`` has no file or there is no record. A record whose
    pass this process owns is answered from that pass, and a settlement a
    failed write left unrecorded is written again. A record that still
    claims running work for an owner proven gone is closed out on disk as
    ``owner_retired`` (see the module docstring). Any other record is
    returned as read: a live owner, or one that cannot be proven exited,
    still holds its work.

    Raises:
        OSError: The record exists but cannot be read.
        ValueError: The record is not a JSON object.
    """
    path = consolidation_drain_record_path(db)
    if path is None:
        return None
    record = read_consolidation_drain_record(path)
    if record is None:
        return None
    pass_id = record.get("pass_id")
    owned = _OWNED_PASSES.get(pass_id) if isinstance(pass_id, str) else None
    if owned is not None:
        if owned.settled:
            owned.write()
        current = dict(owned.record)
        return ConsolidationDrainStatus(
            path=path,
            record=current,
            unsettled=not owned.settled,
            holder=_connection_holder(current, db),
        )
    if record.get("state") not in _UNSETTLED_STATES:
        return ConsolidationDrainStatus(
            path=path, record=record, unsettled=False, holder=None
        )
    proof = _retired_owner_proof(path, record)
    if proof is None:
        return ConsolidationDrainStatus(
            path=path,
            record=record,
            unsettled=True,
            holder=_process_holder(record),
        )
    reconciled = reconciled_consolidation_drain_record(record, proof)
    _publish_reconciliation(path, record, reconciled)
    return ConsolidationDrainStatus(
        path=path, record=reconciled, unsettled=False, holder=None
    )


def _connection_holder(record: Mapping[str, Any], db: Any) -> Optional[str]:
    """``None`` when this process's pass runs on ``db``'s own connection.

    A record that names no connection is taken to be on it: the stronger
    claim when the two cannot be told apart.
    """
    recorded = record.get("connection")
    if recorded is None or recorded == database_connection_generation(db):
        return None
    return f"another connection of this process (PID {os.getpid()})"


def _process_holder(record: Mapping[str, Any]) -> str:
    owner = record.get("owner")
    if not isinstance(owner, dict):
        return (
            "an owner the record does not identify (if no process holds the "
            "database, the record is stale and may be deleted)"
        )
    return (
        f"PID {owner.get('pid')!r} on host {owner.get('host')!r}, which is "
        "running or cannot be proven to have exited (if it is not running, "
        "the record is stale and may be deleted)"
    )


def _retired_owner_proof(path: Path, record: Mapping[str, Any]) -> Optional[str]:
    """Why the owner of ``record``'s work is gone, or ``None`` if it may not be.

    In this process every pass that still holds work is in
    :data:`_OWNED_PASSES`, so a record of this incarnation that none of them
    names holds nothing. Another process's record is closed out only on the
    same positive proof a stale serving record is swept by.
    """
    owner = record.get("owner")
    if not isinstance(owner, dict):
        return None
    if owner.get("runtime_id") == _RUNTIME_ID:
        return "no pass in the process that recorded it still holds the work"
    if recorded_process_exited(path, owner):
        return (
            f"the process that recorded it (PID {owner.get('pid')}) has "
            "exited, and its database work with it"
        )
    return None


def _publish_reconciliation(
    path: Path, record: Mapping[str, Any], reconciled: Mapping[str, Any]
) -> None:
    """Write ``reconciled`` over ``record`` unless a newer record replaced it.

    The re-read narrows, but between processes cannot close, the window in
    which a pass timing out right now could have its fresh record replaced;
    that pass rewrites its record at its next transition. A write that fails
    is logged: the proof holds whether or not it is on disk, and the next
    reader repeats it.
    """
    try:
        current = read_consolidation_drain_record(path)
    except (OSError, ValueError):
        current = None
    if current is None or (
        current.get("pass_id") != record.get("pass_id")
        or current.get("state") != record.get("state")
    ):
        return
    try:
        write_consolidation_drain_record(path, reconciled)
    except OSError:
        logger.warning(
            "Could not persist the reconciliation of memory consolidation "
            "drain record %s",
            path,
            exc_info=True,
        )
        return
    logger.warning(
        "Memory consolidation drain record %s claimed database work %s since "
        "%s, but %s; recorded as settled (%s)",
        path,
        record.get("state"),
        record.get("timed_out_at"),
        reconciled["reconciliation"],
        CLEANUP_OWNER_RETIRED,
    )
