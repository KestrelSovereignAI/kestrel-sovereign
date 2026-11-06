"""Mock storages for strategy-ledger tests that keep no ledger-row record.

The strategy-ledger tool reserves a new row's address and session in the
agent's database before it writes the row, whenever its storage offers
``reserve_strategy_ledger_row`` (#3990), and refuses the row when that fails.
A bare ``MagicMock`` storage answers every attribute, so it looks able to
reserve and then fails on the first await: every row would be refused. A test
whose storage stands in for one with no record and no assertion producer says
so with :func:`without_ledger_row_record`.
"""

from __future__ import annotations

from typing import Any


def without_ledger_row_record(storage: Any) -> Any:
    """Make a mock ``storage`` one that keeps no ledger-row record; returns it."""
    storage.reserve_strategy_ledger_row = None
    return storage
