"""Which sessions host code attested each strategy-ledger row's content in (#3990).

A strategy-ledger row's semantic source names the session of the turn that
recorded the row (#3875), so a governed corpus that excludes the session holds
the row out. ``STRATEGY_LEDGER.yaml`` lives in the agent's own data directory,
so nothing in that file, neither a field naming a session nor the row's ``id``,
can say where a row came from: an agent that can write the file can rewrite
either.

This table is the record the agent cannot write. When a strategy-ledger tool
writes a row, host code first attests the row's exact content, as
``(section, content digest)``, together with the session the turn's rows are
filed under, or no session for a sessionless turn. The attestation is written
before the row reaches the file, so a row the tool published always has one.

The record is an allow-list keyed by content, never by address:

* **Append-only.** An attestation is never updated or deleted, and one digest
  may carry several sessions. Writers therefore cannot erase one another's
  attestations, and an attestation for content that never reached the file
  matches nothing.
* **Content, not address.** The digest covers the row's id and every content
  field (``ledger_assertions.row_content_digest``). A row whose text or id was
  edited by hand has a digest nothing attested, so it is unattested, exactly
  like a row recorded before this record existed.

Projection names the sessions attested for a row's current digest, and a
governed corpus with session exclusions holds out a ledger-derived assertion
that is unattested or attested in an excluded session; see
:meth:`AsyncStorage.assertion_session_attestations`.

Written in every privacy mode, as the ledger file itself is: a row recorded
while semantic writes are denied is projected once they are allowed again, and
its attestation must exist then. The record holds digests and session keys,
never row text. Every query is filtered by ``agent_id``, so a shared backend
cannot read one agent's attestations for another's rows.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

TABLE = "strategy_ledger_row_attestations"

#: How a sessionless attestation is stored. Never a session key: a turn with an
#: empty session key has no session (``filed_session_id`` returns ``None``).
NO_SESSION = ""

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

#: ``(section, content digest)`` -> every session attested for that content,
#: ``None`` standing for a turn with no session. Content absent from the
#: mapping was never attested.
RowAttestations = Mapping[Tuple[str, str], FrozenSet[Optional[str]]]


def _require_agent(agent_id: Optional[str]) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("strategy-ledger row attestations are recorded per agent")


def _require_key(section: str, content_digest: str) -> None:
    if not isinstance(section, str) or not section:
        raise ValueError("a strategy-ledger row attestation needs its section")
    if not isinstance(content_digest, str) or not _DIGEST_RE.fullmatch(content_digest):
        raise ValueError(
            "a strategy-ledger row attestation needs the row's sha256 content digest"
        )


async def attest_row(
    db,
    *,
    agent_id: str,
    section: str,
    content_digest: str,
    session_id: Optional[str],
) -> None:
    """Attest that a row with this content is being recorded in ``session_id``.

    ``None`` attests a turn with no session. Idempotent, and never replaces
    anything: an attestation already recorded for the same content and session
    is left as it is, and one for another session is kept beside it.
    """
    _require_agent(agent_id)
    _require_key(section, content_digest)
    if session_id is not None and (
        not isinstance(session_id, str) or session_id == NO_SESSION
    ):
        raise ValueError("an attested session must be a non-empty key or None")
    await db.execute(
        f"INSERT INTO {TABLE} "
        "(agent_id, section, content_digest, session_id, attested_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (agent_id, section, content_digest, session_id) DO NOTHING",
        (
            agent_id,
            section,
            content_digest,
            NO_SESSION if session_id is None else session_id,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


async def row_attestations(db, *, agent_id: str) -> RowAttestations:
    """Every attestation recorded for one agent's ledger rows."""
    _require_agent(agent_id)
    rows = await db.fetchall(
        f"SELECT section, content_digest, session_id FROM {TABLE} "
        "WHERE agent_id = ?",
        (agent_id,),
    )
    attested: Dict[Tuple[str, str], set] = {}
    for section, content_digest, session_id in rows:
        session = str(session_id)
        attested.setdefault((str(section), str(content_digest)), set()).add(
            None if session == NO_SESSION else session
        )
    return {key: frozenset(sessions) for key, sessions in attested.items()}
