"""Durable evidence that a timed-out consolidation pass left database work running (#2978).

The memory-consolidation deadline cancels its pass, but cancelling an
``aiosqlite`` await does not stop a statement already running on the
connection's worker thread. The backend hands that statement's rollback to a
retained drain and fences the connection behind it, and
``MemorySystem.consolidate`` keeps the MEMORY lock until the drain proves the
statement has returned, or isolates the connection when it cannot wait any
longer.

While that is happening the agent database itself is fenced, so the evidence
cannot be written there. Each timed-out pass writes its own small JSON record
in a directory beside the SQLite file
(``<database>.consolidation-drains/<pass_id>.json``), replaced atomically:

* ``draining`` -- the pass timed out with database work still running, and the
  MEMORY lock is held while it finishes;
* ``isolated`` -- the lock was released before the work finished, so the
  connection fails every operation closed until it does (``isolation`` says
  why: the cleanup deadline ran out, the caller cancelled the wait, the wait
  itself failed, or the connection was closed under the work);
* ``done`` -- the work has settled; ``cleanup`` says how (``finished``,
  ``failed``, or ``retired`` when the connection was closed under it). A
  ``retired`` record is written only once the connection's worker thread,
  which the work ran on, has exited: closing the connection does not stop a
  statement already running there, so until then the record stays
  unsettled, whatever became of the close.

A record has exactly one owner: the pass that wrote it, in the process its
``owner`` field names (PID, start instant, host and PID namespace, as a
serving record does, plus this process's incarnation id), on the connection
generation its ``connection`` field names. Only that pass ever writes the
record, and it writes no other: two passes, on two connections or in two
processes, can never overwrite each other's evidence.

:func:`reconcile_consolidation_drains` gathers every record for a database,
for the ``memory_system`` health check and for ``MemorySystem.initialize``.
It answers each pass this process owns from the pass itself, and that pass
retries a settlement it could not write. It never writes any other record. A
record still claiming running work whose owner has provably exited (its
worker thread, and the statement, went with it) is reported as such, not
cleared: an operator reviews and deletes it. A forked child is a new owner:
the passes it inherits stay its parent's, and it cannot write their records.

The record holds no memory content: identifiers, timestamps and states only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from kestrel_sovereign.audit_time import utc_now_iso
from kestrel_sovereign.multi_agent.liveness import (
    process_identity,
    recorded_process_exited,
)

from .db.sqlite import CANCELLATION_CLEANUP_FINISHED

logger = logging.getLogger(__name__)

#: Appended to the SQLite file's name to name the directory of its records.
DRAIN_RECORD_DIR_SUFFIX = ".consolidation-drains"

#: The single shared record an earlier shape of #2978 kept per database. Never
#: written; read as one more record if one is found.
LEGACY_DRAIN_RECORD_SUFFIX = ".consolidation-drain.json"

DRAIN_STATE_DRAINING = "draining"
DRAIN_STATE_ISOLATED = "isolated"
DRAIN_STATE_DONE = "done"

#: The states that claim database work is still running.
_UNSETTLED_STATES = frozenset({DRAIN_STATE_DRAINING, DRAIN_STATE_ISOLATED})

#: Why a connection was isolated while its abandoned work was still running.
ISOLATION_CLEANUP_DEADLINE = "cleanup_deadline"
ISOLATION_CALLER_CANCELLED = "caller_cancelled"
ISOLATION_WAIT_FAILED = "wait_failed"
ISOLATION_CONNECTION_CLOSED = "connection_closed"

#: Who holds the work a record describes, relative to the connection it was
#: read for. Every kind but ``settled`` is an unsettled record.
HOLDER_SETTLED = "settled"
#: A pass of this process, on the very connection the records were read for.
HOLDER_THIS_CONNECTION = "this_connection"
#: A pass of this process on a connection the database handle the records
#: were read for has since closed, whose worker thread is still running the
#: work. The handle cannot reconnect until that thread exits.
HOLDER_CLOSED_CONNECTION = "closed_connection"
#: A pass of this process, on another connection to the same database.
HOLDER_OTHER_CONNECTION = "other_connection"
#: A pass of this process whose work settled, but whose record could not be
#: told so: on disk it still claims running work. The pass retries the write.
HOLDER_SETTLED_UNRECORDED = "settled_unrecorded"
#: Another process that is running or cannot be proven to have exited.
HOLDER_OTHER_PROCESS = "other_process"
#: Another process that has provably exited without settling the record.
HOLDER_OWNER_EXITED = "owner_exited"
#: Written by this process, but no pass in it holds the record any more.
HOLDER_UNACCOUNTED = "unaccounted"
#: A state this module never writes.
HOLDER_UNKNOWN_STATE = "unknown_state"

_RECORD_VERSION = 1

#: A record is a few hundred bytes; anything far larger is not one.
_MAX_RECORD_BYTES = 64 * 1024

_RECORD_NAME = re.compile(r"^([0-9a-f]{32})\.json$")

#: This process's incarnation. Tells its own records from those of an earlier
#: process that ran under the same PID, whatever its start time reads as.
_RUNTIME_ID = uuid.uuid4().hex

#: Passes this process owns whose settlement is not yet durably recorded, by
#: pass id. A pass is entered when it first writes its record and leaves once
#: its ``done`` state is on disk, so its in-memory record outranks a file a
#: failed write left behind (or a file it never managed to write).
_OWNED_PASSES: Dict[str, "ConsolidationDrain"] = {}


def _forget_parent_passes() -> None:
    """A forked child is a new incarnation that owns none of its parent's passes.

    The parent's aiosqlite worker threads do not survive the fork, and its
    passes are still the parent's to settle.
    """
    global _RUNTIME_ID
    _RUNTIME_ID = uuid.uuid4().hex
    _OWNED_PASSES.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_parent_passes)


def _database_file(db: Any) -> Optional[str]:
    """``db``'s SQLite file, or ``None`` when it has none.

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
    return database


def consolidation_drain_record_dir(db: Any) -> Optional[Path]:
    """The directory of ``db``'s drain records, or ``None`` when it has no file."""
    database = _database_file(db)
    if database is None:
        return None
    path = Path(database)
    return path.with_name(path.name + DRAIN_RECORD_DIR_SUFFIX)


def legacy_consolidation_drain_record_path(db: Any) -> Optional[Path]:
    """Where the earlier single shared record for ``db`` would be."""
    database = _database_file(db)
    if database is None:
        return None
    path = Path(database)
    return path.with_name(path.name + LEGACY_DRAIN_RECORD_SUFFIX)


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


def write_consolidation_drain_record(
    path: Path, record: Mapping[str, Any]
) -> None:
    """Replace the record at ``path`` with ``record``, durably.

    Creates the record directory (owner-only) if it is missing. Written aside,
    fsynced, renamed over the old record and the directory fsynced, so a
    reader sees the previous record or this one, never part of either, and a
    crash cannot lose a state that was reported written.

    Raises:
        OSError: The record could not be written.
    """
    directory = path.parent
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    if not stat.S_ISDIR(os.lstat(directory).st_mode):
        raise NotADirectoryError(
            f"{directory} is not a directory, so it cannot hold drain records"
        )
    staging = directory / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
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
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def read_consolidation_drain_record(path: Path) -> Optional[Dict[str, Any]]:
    """The record at ``path``, or ``None`` when there is none.

    A link is refused rather than followed, and so is anything that is not a
    regular file (opened without blocking, so a FIFO cannot stall the
    reader) or is too large to be a record.

    Raises:
        OSError: The record exists but cannot be read.
        ValueError: The record is not a JSON object.
    """
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"{path} is not a regular file")
        raw = handle.read(_MAX_RECORD_BYTES + 1)
    if len(raw) > _MAX_RECORD_BYTES:
        raise ValueError(f"{path} is larger than any drain record")
    try:
        record = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 text") from exc
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
    The record lives at its own path, named by its pass id, and this object
    is the only writer of it. A write that fails is reported, never raised:
    raising would replace the deadline outcome the caller must see, and the
    settlement it records happens either way. Until its settlement is on disk
    the pass stays in this process's :data:`_OWNED_PASSES`, so a reader here
    trusts it over the file, and :func:`reconcile_consolidation_drains`
    retries the write. Only the process that created the pass writes it: in
    a child forked from that process, :meth:`write` refuses.
    """

    def __init__(
        self, db: Any, *, agent_id: str, deadline_seconds: float
    ) -> None:
        self.record = new_consolidation_drain_record(
            agent_id=agent_id,
            database=_database_file(db),
            connection=database_connection_generation(db),
            deadline_seconds=deadline_seconds,
        )
        #: The backend whose connection runs the work. Its later connections
        #: have other generations, so the pass can tell when its own
        #: connection has been closed under the work.
        self.backend: Any = getattr(db, "backend", None)
        directory = consolidation_drain_record_dir(db)
        self.path: Optional[Path] = (
            None if directory is None else directory / f"{self.pass_id}.json"
        )
        self.settled = False
        #: The state the record on disk holds, or None if no write succeeded.
        self.persisted_state: Optional[str] = None

    @property
    def pass_id(self) -> str:
        return self.record["pass_id"]

    @property
    def owned_here(self) -> bool:
        """Whether this process created the pass (not a child forked from it)."""
        return self.record["owner"].get("runtime_id") == _RUNTIME_ID

    def write(self) -> bool:
        """Persist the current state; whether it reached the disk."""
        if self.path is None:
            logger.warning(
                "No durable location for the memory consolidation drain "
                "record (database has no file); state is %s",
                self.record["state"],
            )
            return False
        if not self.owned_here:
            logger.warning(
                "Not writing memory consolidation drain record %s: its pass "
                "belongs to the process this one was forked from",
                self.path,
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
        self.persisted_state = self.record["state"]
        if self.settled:
            _OWNED_PASSES.pop(self.pass_id, None)
        return True

    def isolate(self, reason: str) -> None:
        """The lock is being released while the work still runs."""
        if self.settled or not self.owned_here:
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
        if self.settled or not self.owned_here:
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
    """One drain record, as reconciled against the owner of its work.

    ``holder`` is one of the ``HOLDER_*`` kinds; every kind but
    :data:`HOLDER_SETTLED` means the record on disk does not say its work
    settled. ``recorded_state`` is the state the file holds: for a pass of
    this process it may lag the pass, and it is None when no write of the
    pass ever succeeded. ``owner_note`` adds what the holder kind alone does
    not say (which process, and how its exit was proven). ``record_missing``
    is True for a pass of this process whose file is not in the directory.
    """

    path: Path
    record: Dict[str, Any]
    holder: str
    recorded_state: Any
    owner_note: Optional[str] = None
    record_missing: bool = False

    @property
    def unsettled(self) -> bool:
        return self.holder != HOLDER_SETTLED

    @property
    def state(self) -> Any:
        return self.record.get("state")

    def _where(self) -> str:
        if self.recorded_state is None:
            return f"(its record could not be written durably: {self.path})"
        if self.record_missing:
            return f"(its record is missing: {self.path})"
        if self.recorded_state != self.state:
            return f"(record: {self.path}, which still says {self.recorded_state})"
        return f"(record: {self.path})"

    def describe(self) -> str:
        """One sentence for an operator: what the record says, and who holds it."""
        timed_out = (
            "A memory consolidation pass timed out at "
            f"{self.record.get('timed_out_at')}"
        )
        where = self._where()
        if self.holder == HOLDER_SETTLED:
            return (
                f"{timed_out}; the database work it left running settled as "
                f"{self.record.get('cleanup')!r} {where}"
            )
        if self.holder == HOLDER_SETTLED_UNRECORDED:
            return (
                f"{timed_out}; the database work it left running settled as "
                f"{self.record.get('cleanup')!r}, but writing that to its "
                "record failed; this process retries on every check "
                f"{where}"
            )
        if self.holder == HOLDER_THIS_CONNECTION:
            if self.state == DRAIN_STATE_DRAINING:
                return (
                    f"{timed_out} and its database work is still running; the "
                    f"memory lock is held until it finishes {where}"
                )
            return (
                f"{timed_out} and its database work is still running; the "
                f"database connection refuses all work until it finishes "
                f"{where}"
            )
        if self.holder == HOLDER_UNKNOWN_STATE:
            return (
                f"{timed_out} and recorded an unknown state {self.state!r}; "
                f"whether its database work settled cannot be told {where}"
            )
        if self.holder == HOLDER_CLOSED_CONNECTION:
            return (
                f"{timed_out} and its database work is still running on a "
                "connection this agent's database has since closed; the work "
                "ends only when that connection's worker thread exits, and "
                f"the database cannot be reconnected until it does {where}"
            )
        if self.holder == HOLDER_OTHER_CONNECTION:
            return (
                f"{timed_out} and its database work is still running "
                f"({self.state}) on another connection of this process "
                f"(PID {os.getpid()}); this agent's database connection is not "
                f"the one holding it, but its writes may wait behind that work "
                f"{where}"
            )
        recorded = f"{timed_out} and recorded its database work as {self.state}"
        if self.holder == HOLDER_OTHER_PROCESS:
            return (
                f"{recorded}, held by {self.owner_note}; this agent's database "
                "connection is not the one holding it, but its writes may wait "
                "behind that work (if that process is not running, the record "
                f"is stale and may be deleted) {where}"
            )
        if self.holder == HOLDER_OWNER_EXITED:
            return (
                f"{recorded}, but {self.owner_note}, so that work ended with "
                "it; nothing settled the record. Review it, then delete it to "
                f"clear this report {where}"
            )
        return (
            f"{recorded}, but no pass in this process holds it any more; "
            "nothing settled the record. Review it, then delete it to clear "
            f"this report {where}"
        )


@dataclass(frozen=True)
class UnreadableConsolidationDrainRecord:
    """A record, or the record directory, that cannot be read."""

    path: Path
    error: str
    is_directory: bool = False

    def describe(self) -> str:
        if self.is_directory:
            return (
                "Could not list the memory consolidation drain records in "
                f"{self.path}: {self.error}"
            )
        return (
            f"Could not read the memory consolidation drain record {self.path}: "
            f"{self.error}"
        )


@dataclass(frozen=True)
class ConsolidationDrainReport:
    """Every drain record for one database, read for one of its connections."""

    directory: Path
    statuses: Tuple[ConsolidationDrainStatus, ...]
    unreadable: Tuple[UnreadableConsolidationDrainRecord, ...]

    @property
    def unsettled(self) -> Tuple[ConsolidationDrainStatus, ...]:
        return tuple(status for status in self.statuses if status.unsettled)

    @property
    def latest_settled(self) -> Optional[ConsolidationDrainStatus]:
        """The most recently timed-out pass whose work settled, as history."""
        settled = [status for status in self.statuses if not status.unsettled]
        if not settled:
            return None
        return max(settled, key=lambda status: str(status.record.get("timed_out_at")))


def reconcile_consolidation_drains(db: Any) -> Optional[ConsolidationDrainReport]:
    """Every drain record for ``db``'s database, each matched to its owner.

    ``None`` when ``db`` has no file. Covers every record in the directory,
    the legacy shared record if one exists, and every pass of this process
    whose record is missing. A pass this process owns is answered from the
    pass itself, never from its file, and a settlement a failed write left
    unrecorded is written again, by that pass, to that pass's own record.
    Nothing else is written: a record of another connection or process is
    reported as read, whether or not its owner is still alive.

    Never raises for the state of the files: a record, or the directory,
    that cannot be read is reported in ``unreadable``, and this process's
    passes are reported all the same.
    """
    directory = consolidation_drain_record_dir(db)
    if directory is None:
        return None
    generation = database_connection_generation(db)
    backend = getattr(db, "backend", None)
    statuses = []
    unreadable = []
    answered = set()
    to_read = []
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        names = []
    except OSError as exc:
        names = []
        unreadable.append(
            UnreadableConsolidationDrainRecord(directory, str(exc), is_directory=True)
        )
    for name in names:
        # Staging files (dot-prefixed) and anything else not named as a
        # record are not records.
        match = _RECORD_NAME.match(name)
        if match is None:
            continue
        path = directory / name
        # A pass id names one pass anywhere, so its file here is that pass's.
        owned = _OWNED_PASSES.get(match.group(1))
        if owned is not None:
            answered.add(owned.pass_id)
            statuses.append(_owned_status(path, owned, generation, backend))
        else:
            to_read.append(path)
    legacy = legacy_consolidation_drain_record_path(db)
    if legacy is not None and os.path.lexists(legacy):
        to_read.append(legacy)

    for path in to_read:
        try:
            record = read_consolidation_drain_record(path)
        except (OSError, ValueError) as exc:
            unreadable.append(UnreadableConsolidationDrainRecord(path, str(exc)))
            continue
        if record is None:
            # Gone between listing and reading: this module never deletes a
            # record, so an operator removed it.
            continue
        pass_id = record.get("pass_id")
        if path.parent == directory and path.name != f"{pass_id}.json":
            unreadable.append(
                UnreadableConsolidationDrainRecord(
                    path, f"it names pass {pass_id!r}, not the one its name does"
                )
            )
            continue
        statuses.append(_recorded_status(path, record))

    for pass_id, owned in list(_OWNED_PASSES.items()):
        if (
            pass_id not in answered
            and owned.path is not None
            and owned.path.parent == directory
        ):
            statuses.append(
                _owned_status(
                    owned.path, owned, generation, backend, record_missing=True
                )
            )
    return ConsolidationDrainReport(
        directory=directory,
        statuses=tuple(statuses),
        unreadable=tuple(unreadable),
    )


def _owned_status(
    path: Path,
    owned: ConsolidationDrain,
    generation: Optional[str],
    backend: Any,
    *,
    record_missing: bool = False,
) -> ConsolidationDrainStatus:
    """The pass of this process whose record is ``path``, answered from the pass.

    ``generation`` and ``backend`` are those of the database the records are
    read for. A backend holds one connection at a time and names each with a
    new generation, so a pass recorded on that same backend under another
    generation ran on a connection the backend has since closed.
    """
    record = dict(owned.record)
    if owned.settled:
        # Still registered only while its settlement is not on disk.
        if owned.pass_id in _OWNED_PASSES and not owned.write():
            return ConsolidationDrainStatus(
                path=path,
                record=record,
                holder=HOLDER_SETTLED_UNRECORDED,
                recorded_state=owned.persisted_state,
                record_missing=record_missing,
            )
        return ConsolidationDrainStatus(
            path=path,
            record=record,
            holder=HOLDER_SETTLED,
            recorded_state=owned.persisted_state,
        )
    recorded = record.get("connection")
    # A record that names no connection is taken to be on this one: the
    # stronger claim when the two cannot be told apart.
    if recorded is None or recorded == generation:
        holder = HOLDER_THIS_CONNECTION
    elif owned.backend is not None and owned.backend is backend:
        holder = HOLDER_CLOSED_CONNECTION
    else:
        holder = HOLDER_OTHER_CONNECTION
    return ConsolidationDrainStatus(
        path=path,
        record=record,
        holder=holder,
        recorded_state=owned.persisted_state,
        record_missing=record_missing,
    )


def _recorded_status(path: Path, record: Dict[str, Any]) -> ConsolidationDrainStatus:
    """A record no pass of this process holds, judged from what it says."""
    state = record.get("state")

    def status(holder: str, owner_note: Optional[str] = None) -> ConsolidationDrainStatus:
        return ConsolidationDrainStatus(
            path=path,
            record=record,
            holder=holder,
            recorded_state=state,
            owner_note=owner_note,
        )

    if state == DRAIN_STATE_DONE:
        return status(HOLDER_SETTLED)
    if state not in _UNSETTLED_STATES:
        return status(HOLDER_UNKNOWN_STATE)
    owner = record.get("owner")
    if not isinstance(owner, dict):
        return status(HOLDER_OTHER_PROCESS, "an owner the record does not identify")
    if owner.get("runtime_id") == _RUNTIME_ID:
        # Every pass of this process that still holds work is in
        # _OWNED_PASSES, so this record's work is not held here.
        return status(HOLDER_UNACCOUNTED)
    if recorded_process_exited(path, owner):
        return status(
            HOLDER_OWNER_EXITED,
            f"the process that owned it (PID {owner.get('pid')}) has "
            "provably exited",
        )
    return status(
        HOLDER_OTHER_PROCESS,
        f"PID {owner.get('pid')!r} on host {owner.get('host')!r}, which is "
        "running or cannot be proven to have exited",
    )
