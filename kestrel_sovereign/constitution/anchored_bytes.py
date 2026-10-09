"""Read the constitution bytes an agent is anchored to.

One question, asked the same way by both reanchor entry points: *what is
actually stored under this agent's ``constitution_hash``, and if we cannot
show it, is that because nothing is there or because we cannot read it?*

The two answers are not interchangeable. **Absent** is the #2616 dangling
anchor a reanchor exists to repair — refusing it would brick the fix.
**Present but unreadable** could be hiding an active Amendment VIII, and an
irrevocable right whose precondition cannot be checked is not a right that may
be waived by accident (#2465).

The read is deliberately **unbound**. ``AsyncFileStore`` scopes an ordinary
read to ``file_owners``, and a row with no ownership entry comes back as
``None`` — indistinguishable from no row at all. That is not a corner case
here: ``file_owners`` arrived with #2649, every agent in the pre-#1118 cohort
this guard protects stored its constitution before that, and the backfill only
claims a blob when the agent carries a ``governed_by`` edge whose target equals
its ``constitution_hash`` — precisely the edge that has drifted in the #2616
population. Scoping this read would report "absent" for a constitution sitting
in the table byte-for-byte, and the guard would permit the erasure it exists to
prevent.

Reading it unbound is safe for the same reason it is necessary: the hash comes
from *this* agent's own node, and the store is content-addressed, so any row
under that hash holds exactly those bytes. Same argument as the unscoped
``governed_by`` read in ``constitution_reanchor._read_agent_anchor``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


def historical_anchor_hash(
    properties: Mapping, governed_by_targets: Iterable[str],
) -> Optional[str]:
    """Recover evidence, never authority, when the operative pointer is lost.

    Native current/history receipts survive edge/pointer deletion. Inspect
    their typed hash fields, not arbitrary receipt prose. Without the pointer
    conflicting, malformed or excessive evidence must fail closed; an
    operator can restore the exact prior pointer before attempting repair.
    An intact pointer remains authoritative: old receipt history normally
    names multiple superseded constitutions and is not a competing pointer.
    """
    pointer = properties.get("constitution_hash")
    if pointer:
        return pointer
    candidates: set[str] = set()

    def add(value, *, absent_ok=True):
        if value is None and absent_ok:
            return
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
        candidates.add(value)

    targets = tuple(governed_by_targets)
    if len(targets) > 128:
        raise ValueError("Missing anchor pointer has excessive historical governance evidence")
    for target in targets:
        add(target, absent_ok=False)

    def inspect(receipt, kind):
        if not isinstance(receipt, Mapping):
            raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
        fields = ("constitution_hash",) if kind == "genesis_audit" else ("old_hash", "new_hash")
        found = False
        for field in fields:
            value = receipt.get(field)
            if value is None or (field == "old_hash" and value == "none"):
                continue
            add(value)
            found = True
        if not found:
            raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")

    for kind in ("genesis_audit", "constitution_reanchor"):
        current = properties.get(kind)
        if current is not None:
            inspect(current, kind)
        history = properties.get(kind + "_history")
        if history is None:
            continue
        if not isinstance(history, list) or len(history) > 128:
            raise ValueError("Missing anchor pointer has unreadable or excessive historical governance receipt evidence")
        for entry in history:
            if not isinstance(entry, Mapping):
                raise ValueError("Missing anchor pointer has unreadable historical governance receipt evidence")
            inspect(entry.get("receipt"), kind)
            add(entry.get("superseded_by_constitution_hash"))
    if len(candidates) > 1:
        raise ValueError("Missing anchor pointer has ambiguous historical governance; restore its exact prior pointer before signed repair")
    return next(iter(candidates), None)


async def read_anchored_constitution(
    db: "AsyncDatabase", anchored_hash: str
) -> Tuple[Optional[str], bool]:
    """Return ``(text, present)`` for the blob stored under ``anchored_hash``.

    ``(None, False)`` — nothing is stored under that hash. ABSENT.
    ``(None, True)``  — something is, and this process cannot turn it into
    text: a wrong ``KESTREL_DATA_KEY``, corruption, or bytes that are not
    UTF-8. UNREADABLE.
    ``(text, True)``  — the anchored constitution.
    """
    from kestrel_sovereign.security.encryption import DecryptionError
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore

    # No agent_id: see the module docstring. This is the whole point.
    store = AsyncFileStore(db)
    try:
        raw = await store.retrieve_file(anchored_hash)
    except DecryptionError:
        # UNREADABLE means "the bytes are there and this process cannot open
        # them" — a wrong KESTREL_DATA_KEY. Deliberately narrow: a dropped
        # connection is not a key problem, and swallowing it here would tell
        # the operator to go check their data key. Every caller already has a
        # boundary that names a database failure for what it is, so those
        # propagate to it.
        logger.warning(
            "The constitution stored under %s could not be decrypted",
            anchored_hash[:12],
            exc_info=True,
        )
        return None, True
    if raw is None:
        return None, False
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        logger.warning(
            "The constitution stored under %s is not UTF-8 text",
            anchored_hash[:12],
            exc_info=True,
        )
        return None, True
