"""The session each strategy-ledger row was recorded in, as host code recorded it (#3990).

A strategy-ledger row's semantic source names the session of the turn that
recorded the row (#3875), so a governed corpus that excludes the session holds
the row out. ``STRATEGY_LEDGER.yaml`` names that session in each row's
``session_id`` field, but the file lives in the agent's own data directory: an
agent that can write the file can rewrite or delete the field, or the row's
``id``, before a row is projected, and the row would then escape the
exclusion.

These tables are the record the agent cannot write. They hold addresses and
session keys only, never row text.

``strategy_ledger_row_sessions``
    One row per ledger row address the host knows of. The strategy-ledger tool
    reserves a row's address here, with its session, before the row reaches
    the file (:func:`reserve_row_session`): projection never sees a row the
    tool wrote without its record. A reservation never replaces a record, and
    only the attempt that made one may remove it
    (:func:`release_row_reservation`), so one writer cannot erase another's
    binding. Rows already in the file when this record began are recorded once
    as written in no session (:func:`take_row_inventory`).

``strategy_ledger_row_inventories``
    Whether that inventory has been taken for an agent. Once it has, a row
    with no record is not one recorded before sessions were named: nothing the
    agent cannot write says where it came from.

``strategy_ledger_row_refusals``
    The rows the last projection pass could not verify. The governed corpus
    build refuses while any remain (:func:`unresolved_row_refusals`), and a
    later pass removes them once the file agrees with the record again or the
    row is gone.

Written in every privacy mode, as the ledger file itself is: a row recorded
while semantic writes are denied is projected after they are allowed again,
and its session must still be verifiable then.

Every query is filtered by ``agent_id``, so a shared backend cannot read one
agent's record for another's row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple
from uuid import uuid4

TABLE = "strategy_ledger_row_sessions"
INVENTORY_TABLE = "strategy_ledger_row_inventories"
REFUSALS_TABLE = "strategy_ledger_row_refusals"

#: The ``attempt_id`` of a record the inventory made. No reservation is ever
#: made under it, so no attempt can release one.
INVENTORY_ATTEMPT = "inventory"

#: How many addresses one reservation tries before giving up: each retry means
#: another writer reserved the address just read as free.
RESERVATION_ATTEMPTS = 4

#: ``(section, row_id)`` -> the recorded session, ``None`` for a row recorded
#: in no session. A row absent from the mapping was never recorded here.
RowSessions = Dict[Tuple[str, str], Optional[str]]


class RowReservationContended(RuntimeError):
    """Every address tried was reserved by another writer first."""


@dataclass(frozen=True)
class RowReservation:
    """An address reserved, with its session, for one row about to be written."""

    section: str
    row_id: str
    session_id: Optional[str]
    #: Owns the reservation: only the attempt holding it may release it.
    attempt_id: str


def _require_agent(agent_id: Optional[str]) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("strategy-ledger row sessions are recorded per agent")


def _require_address(section: str, row_id: str) -> None:
    if not isinstance(section, str) or not section:
        raise ValueError("a strategy-ledger row session needs its section")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("a strategy-ledger row session needs its row id")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def recorded_row_ids(db, *, agent_id: str, section: str) -> FrozenSet[str]:
    """Every address recorded for one agent's section, row or reservation."""
    _require_agent(agent_id)
    rows = await db.fetchall(
        f"SELECT row_id FROM {TABLE} WHERE agent_id = ? AND section = ?",
        (agent_id, section),
    )
    return frozenset(str(row_id) for (row_id,) in rows)


async def reserve_row_session(
    db,
    *,
    agent_id: str,
    section: str,
    session_id: Optional[str],
    mint: Callable[[FrozenSet[str]], str],
) -> RowReservation:
    """Reserve an address no record holds, with the session its row is recorded in.

    ``mint`` is given every address already recorded for the section and
    returns the address the new row takes, which must not be one of them. The
    reservation is inserted only where no record exists, never over one: a
    writer whose view of the file is stale must not replace the session of a
    row another writer already put there. When another writer reserves the
    same address first, the next one ``mint`` names is tried.

    Raises when no address can be reserved, so the caller writes no row.
    """
    _require_agent(agent_id)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("a strategy-ledger row session must be a string or None")
    attempt_id = uuid4().hex
    for _ in range(RESERVATION_ATTEMPTS):
        recorded = await recorded_row_ids(db, agent_id=agent_id, section=section)
        row_id = mint(recorded)
        _require_address(section, row_id)
        if row_id in recorded:
            raise ValueError(f"{row_id!r} is already a recorded address")
        inserted = await db.execute(
            f"INSERT INTO {TABLE} "
            "(agent_id, section, row_id, session_id, attempt_id, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (agent_id, section, row_id) DO NOTHING",
            (agent_id, section, row_id, session_id, attempt_id, _now()),
        )
        if inserted:
            return RowReservation(
                section=section,
                row_id=row_id,
                session_id=session_id,
                attempt_id=attempt_id,
            )
    raise RowReservationContended(
        f"every address tried for a new {section} row was reserved by another "
        "writer first"
    )


async def release_row_reservation(
    db, *, agent_id: str, reservation: RowReservation
) -> bool:
    """Remove a reservation whose row was not written, if this attempt still owns it.

    Left in place, it would vouch for whichever row later holds the address.
    Removes nothing another attempt recorded. Returns whether it was removed.
    """
    _require_agent(agent_id)
    removed = await db.execute(
        f"DELETE FROM {TABLE} "
        "WHERE agent_id = ? AND section = ? AND row_id = ? AND attempt_id = ?",
        (
            agent_id,
            reservation.section,
            reservation.row_id,
            reservation.attempt_id,
        ),
    )
    return bool(removed)


async def recorded_row_sessions(db, *, agent_id: str) -> RowSessions:
    """Every row session recorded for one agent."""
    _require_agent(agent_id)
    rows = await db.fetchall(
        f"SELECT section, row_id, session_id FROM {TABLE} WHERE agent_id = ?",
        (agent_id,),
    )
    return {
        (str(section), str(row_id)): None if session_id is None else str(session_id)
        for section, row_id, session_id in rows
    }


async def row_inventory_taken(db, *, agent_id: str) -> bool:
    """Whether the rows present when this record began have been recorded."""
    _require_agent(agent_id)
    row = await db.fetchone(
        f"SELECT 1 FROM {INVENTORY_TABLE} WHERE agent_id = ?", (agent_id,)
    )
    return row is not None


async def take_row_inventory(
    db, *, agent_id: str, rows: Iterable[Tuple[str, str]]
) -> bool:
    """Record ``rows`` as written in no session, once per agent.

    ``rows`` are the ``(section, row_id)`` addresses in the ledger, as it
    stands when this record begins, of the rows whose file names no session:
    rows recorded before #3875, and rows written into the file by hand before
    now. Nothing earlier than now can be told apart from them. A row that
    already has a record keeps it. After this, only the strategy-ledger tool
    gives a row a record.

    Returns whether this call took it; ``False`` when it was already taken,
    in which case nothing is written.
    """
    _require_agent(agent_id)
    addresses = sorted(set(rows))
    for section, row_id in addresses:
        _require_address(section, row_id)
    recorded_at = _now()
    async with db.transaction(immediate=True):
        if await row_inventory_taken(db, agent_id=agent_id):
            return False
        if addresses:
            await db.execute_many(
                f"INSERT INTO {TABLE} "
                "(agent_id, section, row_id, session_id, attempt_id, recorded_at) "
                "VALUES (?, ?, ?, NULL, ?, ?) "
                "ON CONFLICT (agent_id, section, row_id) DO NOTHING",
                [
                    (agent_id, section, row_id, INVENTORY_ATTEMPT, recorded_at)
                    for section, row_id in addresses
                ],
            )
        await db.execute(
            f"INSERT INTO {INVENTORY_TABLE} (agent_id, taken_at) VALUES (?, ?) "
            "ON CONFLICT (agent_id) DO NOTHING",
            (agent_id, recorded_at),
        )
    return True


async def replace_row_refusals(
    db,
    *,
    agent_id: str,
    sections: Iterable[str],
    refusals: Mapping[Tuple[str, str], str],
) -> None:
    """Make ``refusals`` the unresolved refusals of ``sections``.

    ``sections`` are the ones a projection pass verified in full. A refusal
    recorded earlier in one of them, for a row the pass found verified or
    gone, is resolved; refusals in any other section are left as they are.
    ``refusals`` maps a row address to why its session cannot be verified.
    """
    _require_agent(agent_id)
    covered = sorted(set(sections))
    stray = {section for section, _ in refusals} - set(covered)
    if stray:
        raise ValueError(f"refusals outside the sections verified: {sorted(stray)}")
    detected_at = _now()
    async with db.transaction():
        for section in covered:
            await db.execute(
                f"DELETE FROM {REFUSALS_TABLE} WHERE agent_id = ? AND section = ?",
                (agent_id, section),
            )
        if refusals:
            await db.execute_many(
                f"INSERT INTO {REFUSALS_TABLE} "
                "(agent_id, section, row_id, reason, detected_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (agent_id, section, row_id, reason, detected_at)
                    for (section, row_id), reason in sorted(refusals.items())
                ],
            )


async def unresolved_row_refusals(db, *, agent_id: str) -> int:
    """How many rows the last projection could not verify."""
    _require_agent(agent_id)
    count = await db.fetchval(
        f"SELECT COUNT(*) FROM {REFUSALS_TABLE} WHERE agent_id = ?", (agent_id,)
    )
    return int(count or 0)
