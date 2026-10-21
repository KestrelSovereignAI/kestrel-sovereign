"""Refuse a test ID too long for the tools that read a CI log (#3669).

``pytest -v`` prints every node ID on a line of its own, and again in the
FAILURES header and the short summary. ``gh run view --log`` reads a job log
with Go's default ``bufio.Scanner`` and ignores its error, so it stops at the
first line over 64 KiB without a word. From #3628, one parametrized test put a
100 KB and a 1 MB payload into its IDs, and every unit-tier log read through
gh ended at ~39% with every test so far passing. Twice that was taken for a
stalled step, while the real failure sat in the unread part of the log.

A node ID is a name. Every ID in the suite was under 420 characters except
four, and each of those four carried its test's payload. The limit refuses
that class with a wide margin under any reader's line limit. Give the case an
explicit ``id``.

Under xdist the controller must refuse too, from the IDs each worker reports:
a ``UsageError`` raised only in a worker never reaches the job log, and the run
ends in an unrelated ``INTERNALERROR`` instead. A worker reports its IDs even
when its collection hook raised, so the controller always sees them.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

MAX_NODE_ID_LENGTH = 1000

#: The error quotes this much of a refused ID, one ID to a line, and at most
#: this many IDs, so no line of the error is too long itself.
_QUOTED = 100
_LISTED = 20


def _refuse(nodeids: Iterable[str]) -> None:
    refused = sorted(nodeid for nodeid in nodeids if len(nodeid) > MAX_NODE_ID_LENGTH)
    if not refused:
        return
    lines = [
        f"  {nodeid[:_QUOTED]!r}... ({len(nodeid)} characters)"
        for nodeid in refused[:_LISTED]
    ]
    if len(refused) > _LISTED:
        lines.append(f"  ... and {len(refused) - _LISTED} more")
    raise pytest.UsageError(
        f"These test IDs are longer than {MAX_NODE_ID_LENGTH} characters. "
        "`gh run view --log` silently stops reading a CI log at a line over "
        "64 KiB, so a payload in an ID hides the rest of the run (#3669). "
        "Give each case an explicit `id`:\n" + "\n".join(lines)
    )


def refuse_overlong_node_ids(items: list[pytest.Item]) -> None:
    """Refuse a run that would print a node ID over the limit.

    Called from ``pytest_collection_modifyitems``. Under xdist the message
    reaches the log from :func:`pytest_xdist_node_collection_finished`.
    """
    _refuse(item.nodeid for item in items)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    """Refuse, on the xdist controller, a worker's collection over the limit."""
    _refuse(ids)
