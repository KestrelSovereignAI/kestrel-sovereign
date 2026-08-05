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
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


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
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore

    # No agent_id: see the module docstring. This is the whole point.
    store = AsyncFileStore(db)
    try:
        raw = await store.retrieve_file(anchored_hash)
    except Exception:  # noqa: BLE001 — stored, but not readable here
        logger.warning(
            "The constitution stored under %s could not be read",
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
