"""Reading the bytes an agent is anchored to (#2465).

Three answers, and the Iron Rule guard treats them differently, so this module
has to keep them apart:

  ABSENT      — nothing stored under that hash. The #2616 dangling anchor a
                reanchor exists to repair; the guard permits it.
  UNREADABLE  — the bytes are there and this process cannot open them. Could be
                hiding an active Amendment VIII; the guard refuses.
  the text    — decide on it.

A *database* failure is none of the three. It used to be swallowed as
UNREADABLE, which fails closed but tells the operator to go check their
``KESTREL_DATA_KEY`` when the real problem is that PostgreSQL went away. Both
callers already have a boundary that names a database failure for what it is,
so those propagate to it.
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.constitution.anchored_bytes import (
    read_anchored_constitution,
)

HASH = "a" * 64


class _Rows:
    """The one query this module issues, answered however the test needs."""

    def __init__(self, row=None, raises=None):
        self._row = row
        self._raises = raises
        self.queries: list[str] = []

    async def fetchone(self, query, params=()):
        self.queries.append(query)
        if self._raises is not None:
            raise self._raises
        return self._row


@pytest.mark.asyncio
async def test_a_missing_row_is_absent():
    text, present = await read_anchored_constitution(_Rows(row=None), HASH)
    assert (text, present) == (None, False)


@pytest.mark.asyncio
async def test_stored_plaintext_comes_back_as_text():
    rows = _Rows(row=(b"# Kestrel Constitution\n", None))
    text, present = await read_anchored_constitution(rows, HASH)
    assert (text, present) == ("# Kestrel Constitution\n", True)


@pytest.mark.asyncio
async def test_the_read_is_not_ownership_scoped():
    """The reason this module exists. A blob with no ``file_owners`` row would
    read back as ABSENT through a bound store — the branch that *permits* the
    reanchor — for exactly the pre-#2649 cohort the guard protects."""
    rows = _Rows(row=(b"bytes", None))
    await read_anchored_constitution(rows, HASH)
    assert rows.queries and "file_owners" not in rows.queries[0]


@pytest.mark.asyncio
async def test_bytes_that_will_not_decrypt_are_unreadable():
    """A wrong ``KESTREL_DATA_KEY``: the row is marked encrypted and no key
    opens it."""
    rows = _Rows(row=(b"\x00not-a-valid-token", json.dumps({"enc": True})))
    text, present = await read_anchored_constitution(rows, HASH)
    assert (text, present) == (None, True)


@pytest.mark.asyncio
async def test_bytes_that_are_not_utf8_are_unreadable():
    rows = _Rows(row=(b"\xff\xfe\x00garbage", None))
    text, present = await read_anchored_constitution(rows, HASH)
    assert (text, present) == (None, True)


@pytest.mark.asyncio
async def test_a_database_failure_is_not_reported_as_a_key_problem():
    """It propagates. Classifying it as UNREADABLE still fails closed, but the
    refusal then says "check KESTREL_DATA_KEY" about a dropped connection, and
    the caller's own boundary — which names the database — never runs."""
    rows = _Rows(raises=OSError("connection reset by peer"))
    with pytest.raises(OSError, match="connection reset by peer"):
        await read_anchored_constitution(rows, HASH)
