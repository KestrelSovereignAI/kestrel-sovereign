"""Who opened a turn, as recorded on the row that opened it (#3654).

Every turn opens with one ``user`` row. The agent writes that row's provenance
into its metadata, and :func:`turn_opener_metadata` is the one spelling every
opening writer uses (the non-streaming and streaming turn bodies and the
bootstrap discovery flow):

* a signal wake carries ``signal_wake = {"source", "mode"}``, as the
  dispatcher hands it to ``process_input`` (#2204). The turn belongs to the
  agent's own autonomous chain.
* every other turn was sent to the agent, by a person in the console or on a
  channel, or by a client acting for one (an API caller, a parent agent
  delegating a task). It carries ``turn_opener = "person"``.

The store reads this back when a turn arrives with no session
(:func:`latest_turn_opened_by_person`): it joins the newest session only when a
person opened that session's latest turn. Rows written before the tag existed
carry neither mark, and neither does anything else that never recorded who
opened its turn. They are not evidence of a person, so the turn starts a fresh
session instead of guessing from timing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple

from kestrel_sovereign.storage.session_grouping import signal_wake_source
from kestrel_sovereign.storage.session_id_column import SESSION_ID_KEY

#: Metadata key a non-wake turn's opening row is stamped with.
TURN_OPENER_KEY = "turn_opener"

#: The value of :data:`TURN_OPENER_KEY` for a turn sent to the agent.
PERSON_OPENER = "person"

#: How many of the agent's newest rows are searched for the row that opened a
#: session's latest turn.
#:
#: Turns are serialized per agent (the turn lifecycle owns CONVERSATION), so
#: the rows newer than a turn's opening row are that turn's own, at most an
#: operator notice and the reply, plus any non-turn rows (a system notice, a
#: compaction marker) written meanwhile. This window is far deeper than that.
#: A session whose opening row lies outside it cannot be classified, and is
#: not joined.
LATEST_TURN_SCAN_ROWS = 50


def turn_opener_metadata(signal_wake: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The provenance a turn's opening ``user`` row is persisted with.

    ``signal_wake`` is what the dispatcher passed to ``process_input`` for a
    signal wake, and ``None`` for a turn that was sent to the agent. Returns a
    new dict on every call, so a caller may add its own keys to it.
    """
    if signal_wake:
        return {"signal_wake": signal_wake}
    return {TURN_OPENER_KEY: PERSON_OPENER}


def opens_turn(role: Optional[str], metadata: Dict[str, Any]) -> bool:
    """Whether a row is the one that opened a turn.

    Every turn opens with one ``user`` row: what was sent, or a signal wake's
    rendered prompt. An operator-signal notice is persisted as ``user`` too, so
    that it replays, but ``inject_operator_turn`` adds it to a turn that has
    already opened, after that turn's own row. It opens nothing; it belongs to
    the turn it was injected into, whoever opened that one.
    """
    return role == "user" and not metadata.get("operator_signal")


def opened_by_person(metadata: Dict[str, Any]) -> bool:
    """Whether an opening row's metadata says a person opened its turn.

    Only the positive tag counts. A ``signal_wake`` tag is an autonomous wake,
    whatever else the row carries. A row with neither mark (written before the
    tag existed, or by a writer that never recorded who opened its turn) is
    not evidence of a person.
    """
    if signal_wake_source(metadata) is not None:
        return False
    return metadata.get(TURN_OPENER_KEY) == PERSON_OPENER


def latest_turn_opened_by_person(
    rows: Iterable[Tuple[Optional[str], Any]],
    session_id: str,
) -> bool:
    """Whether ``session_id``'s latest turn was opened by a person.

    ``rows`` are ``(role, metadata)`` pairs as stored, newest first. The walk
    looks for the row that opened the session's latest turn: replies, system
    rows and operator-signal notices belong to a turn without opening it (see
    :func:`opens_turn`), and rows filed under another session are not this
    session's. The latest row of an autonomous chain is usually the agent's
    reply to a wake, so classifying the literal latest row would misread it.

    The opening row's own metadata decides, through :func:`opened_by_person`.
    Anything else is ``False`` too: a metadata document that cannot be read
    (it may be the opening row, filed under this session), or no opening row
    in ``rows``. Without positive evidence of a person's turn the caller starts
    a fresh session rather than falling back to timing.
    """
    for role, raw in rows:
        if raw is None or raw == "":
            continue  # files under no session, so not this session's opener
        if isinstance(raw, dict):
            metadata = raw
        else:
            try:
                metadata = json.loads(raw)
            except (TypeError, ValueError):
                return False
            if not isinstance(metadata, dict):
                return False
        if metadata.get(SESSION_ID_KEY) != session_id:
            continue
        if not opens_turn(role, metadata):
            continue
        return opened_by_person(metadata)
    return False
