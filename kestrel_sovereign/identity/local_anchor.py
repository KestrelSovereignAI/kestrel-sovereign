"""Read an agent's DID from the local identity anchor.

Identity is born in ``agent_data/<Name>/kestrel_prime.db`` — twelve places
across seven modules read that file's existence as the fact that a directory
*is* an agent — and it stays there on every backend (#2871). Governance lives
in whatever database the runtime resolves. Asking the runtime database who
this agent is inverts that, and on a PostgreSQL host the answer is "nobody"
until boot has replicated the birth record, which is downstream of the
question (#2894).

Extracted from ``multi_agent.agent_manager`` so the process-per-agent server
and the offline governance tools ask it the same way the in-process
multi-agent host already does, instead of each growing its own answer.
"""

from __future__ import annotations

import asyncio
import sqlite3
from enum import Enum
from pathlib import Path


class AgentDIDLookupMode(str, Enum):
    """What the caller is about to do, which decides how the anchor is opened.

    Three intents, because two of them cannot share an answer about a WAL:

    ``COLD_READ_ONLY`` is an authority decision (scheduler discovery). It must
    neither write nor act on uncertain identity state, so live WAL state is a
    refusal.

    ``INITIALIZATION`` is real agent startup. It opens read/write precisely so
    SQLite can recover a legitimately interrupted WAL before the agent's own
    storage opens.

    ``INSPECTION`` reports on an agent it must not disturb. It never writes and
    never checkpoints, but unlike a cold authority read it must not refuse over
    leftover sidecars: an unclean stop leaves those routinely, and the drift
    report is exactly what an operator needs then — `kestrel doctor` prescribes
    the reanchor command that would refuse (#2920).
    """

    COLD_READ_ONLY = "cold_read_only"
    INITIALIZATION = "initialization"
    INSPECTION = "inspection"


class AnchorAbsent(ValueError):
    """This directory holds no birth record — there is nothing here to read.

    Distinguished from every other failure of :func:`read_anchor_agent_did`
    because it is the only one a caller may answer from somewhere else. A
    corrupt file, a permission denial, two agent roots, or live WAL state
    during a cold read are all cases where an anchor **is** present and this
    process could not be told what it says. Treating those as "no anchor" lets
    a caller fall through to another database and adopt a different agent's
    identity — the #2871 rule inverted, since an identity gap must refuse.
    """


async def read_anchor_agent_did(
    storage_dir: str,
    *,
    mode: AgentDIDLookupMode = AgentDIDLookupMode.COLD_READ_ONLY,
) -> str:
    """Read a local agent DID with an explicit cold-vs-startup safety mode.

    ``AsyncStorage.initialize()`` is deliberately write-capable: on a missing
    SQLite path it creates the database, WAL, audit tables, and schema.  Cold
    scheduler discovery is only a lookup and must never turn an unincepted
    configuration entry into a blank identity that blocks later inception.

    ``COLD_READ_ONLY`` uses SQLite immutable read-only mode and refuses WAL
    sidecars, because a scheduler authority decision must never ignore pending
    identity state or alter the cold agent directory. ``INITIALIZATION`` is
    used only by normal agent startup: it still refuses a missing database,
    but opens it read/write so SQLite can replay a legitimate interrupted WAL
    before the real agent storage opens. The connection is created and closed
    in the worker thread so no descriptor survives a lookup failure.
    """
    if not isinstance(mode, AgentDIDLookupMode):
        raise ValueError(f"Unknown agent identity lookup mode: {mode!r}")
    db_path = Path(storage_dir) / "kestrel_prime.db"

    def _lookup() -> str:
        if not db_path.is_file():
            raise AnchorAbsent(
                f"No agent found in {storage_dir}. "
                "Run inception first: kestrel create <name>"
            )

        sidecars = (
            Path(f"{db_path}-wal"),
            Path(f"{db_path}-shm"),
        )
        cold_read_only = mode is AgentDIDLookupMode.COLD_READ_ONLY
        inspection = mode is AgentDIDLookupMode.INSPECTION
        # An inspection reads the WAL rather than ignoring or replaying it, so
        # it only takes the immutable path when there is no WAL to miss. Either
        # way it writes nothing and leaves no file that was not already there.
        inspection_ignores_wal = inspection and not any(
            sidecar.exists() for sidecar in sidecars
        )
        # A normal ``mode=ro`` connection can still create SQLite's shared
        # memory and WAL sidecars when it opens a WAL-mode database.  Besides
        # violating a cold lookup's read-only contract, ``immutable=1`` would
        # ignore a pre-existing WAL and could authorize an old identity.
        # Startup intentionally takes the opposite path: it must let SQLite
        # recover a real WAL after a crash before the agent opens its storage.
        if cold_read_only and any(sidecar.exists() for sidecar in sidecars):
            raise ValueError(
                f"Could not safely read local agent identity from {storage_dir}: "
                "SQLite WAL state is present"
            )

        connection = None
        try:
            # ``Path.as_uri`` handles spaces and platform path escaping. Cold
            # discovery accepts only a checkpointed identity and cannot create
            # sidecars; normal initialization opens an existing DB read/write
            # so SQLite can recover its own WAL state. ``mode=rw`` still
            # refuses an unincepted/missing database.
            if cold_read_only or inspection_ignores_wal:
                uri_flags = "mode=ro&immutable=1"
            elif inspection:
                uri_flags = "mode=ro"
            else:
                uri_flags = "mode=rw"
            connection = sqlite3.connect(
                f"{db_path.resolve().as_uri()}?{uri_flags}",
                uri=True,
            )
            rows = connection.execute(
                "SELECT node_id FROM graph_nodes "
                "WHERE node_type = ? ORDER BY node_id",
                ("agent",),
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(
                f"Could not read local agent identity from {storage_dir}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

        # Catch a writer that raced the pre-open cold sidecar check. An
        # immutable reader deliberately ignores WAL data, so returning a DID
        # after this transition would be a stale-identity authorization
        # decision. Startup has deliberately consumed the normal SQLite path.
        if (cold_read_only or inspection_ignores_wal) and any(
            sidecar.exists() for sidecar in sidecars
        ):
            raise ValueError(
                f"Could not safely read local agent identity from {storage_dir}: "
                "SQLite WAL state appeared during lookup"
            )

        if not rows:
            raise AnchorAbsent(
                f"No agent found in {storage_dir}. "
                "Run inception first: kestrel create <name>"
            )
        if len(rows) != 1 or not isinstance(rows[0][0], str) or not rows[0][0]:
            # Multiple roots are an identity-integrity failure.  Picking one by
            # incidental SQLite order would authorize the wrong tenant.
            raise ValueError(
                f"Local identity database in {storage_dir} has an invalid "
                "agent root set"
            )
        return rows[0][0]

    return await asyncio.to_thread(_lookup)
