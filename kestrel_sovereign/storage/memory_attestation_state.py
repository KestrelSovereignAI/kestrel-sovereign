"""How far an agent's sleep attestation has reached (#3544).

The memory feature's pre-sleep hook judges the memories retrieved since its
last completed pass, and stores where that pass's window ended: the agent's
watermark. Consolidation reads it too. A retrieval at or after it is still
waiting to be judged, so it is not archived until it has been.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

#: One row per agent: the instant up to which retrievals have been attested.
ATTESTATION_STATE_TABLE = "memory_attestation_state"
#: The retrievals counted applied, kept until the watermark passes them, so
#: neither a retry nor an overlapping pass counts one retrieval twice.
ATTESTATION_MARKS_TABLE = "memory_attestation_marks"
#: Fixed width, so the stored text sorts as the instants do.
_INSTANT_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"

_TABLES = (
    (
        ATTESTATION_STATE_TABLE,
        f"""CREATE TABLE IF NOT EXISTS {ATTESTATION_STATE_TABLE} (
            agent_id TEXT PRIMARY KEY,
            attested_through TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
    ),
    (
        ATTESTATION_MARKS_TABLE,
        f"""CREATE TABLE IF NOT EXISTS {ATTESTATION_MARKS_TABLE} (
            agent_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            retrieved_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, message_id, retrieved_at)
        )""",
    ),
)


def format_instant(instant: datetime) -> str:
    """``instant`` as the attestation tables store it."""
    return instant.astimezone(timezone.utc).strftime(_INSTANT_FORMAT)


def last_retrieved(metadata: Dict[str, Any]) -> Optional[datetime]:
    """A memory's latest retrieval: its ``last_accessed``, in UTC.

    ``None`` when it was never retrieved or the stamp does not parse. A naive
    stamp is UTC.
    """
    value = metadata.get("last_accessed")
    if isinstance(value, datetime):
        instant = value
    elif isinstance(value, str) and value:
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def awaiting_attestation(
    metadata: Dict[str, Any], watermark: Optional[datetime]
) -> bool:
    """Whether the memory's latest retrieval is still to be judged.

    It is when it falls at or after ``watermark``: no completed pass has
    covered it. With no watermark no pass has begun, and nothing waits.
    """
    if watermark is None:
        return False
    retrieved = last_retrieved(metadata)
    return retrieved is not None and retrieved >= watermark


async def ensure_attestation_tables(db: Any) -> None:
    """Create the attestation tables, safely under concurrent first use.

    Probe, take the migration lock, probe again: two PostgreSQL sessions that
    both run a bare ``CREATE TABLE IF NOT EXISTS`` can both see the table
    missing, and one then fails on ``pg_class``'s unique index.
    """
    for table, ddl in _TABLES:
        if not await db.table_exists(table):
            async with db.migration_lock(f"create_{table}"):
                if not await db.table_exists(table):
                    await db.execute(ddl)
        if not await db.table_exists(table):
            raise RuntimeError(f"{table} was not created")


async def attested_through(db: Any, agent_id: str) -> Optional[datetime]:
    """The agent's watermark, or ``None`` when no pass has begun for it.

    A read only: an agent whose hook never ran has no table to read.
    """
    if not await db.table_exists(ATTESTATION_STATE_TABLE):
        return None
    row = await db.fetchone(
        f"SELECT attested_through FROM {ATTESTATION_STATE_TABLE} "
        "WHERE agent_id = ?",
        (agent_id,),
    )
    # Written only by ``format_instant``: anything else is corruption, and
    # guessing would attest, or archive, the wrong memories.
    return None if row is None else datetime.fromisoformat(row[0])
