"""Mock storages for strategy-ledger tests that keep no row attestations.

The strategy-ledger tools attest a new row's content and session in the
agent's database before they write the row, whenever their storage offers
``attest_strategy_ledger_row`` (#3990), and refuse the row when that fails. A
bare ``MagicMock`` storage answers every attribute, so it looks able to attest
and then fails on the first await: every row would be refused. A test whose
storage stands in for one with no attestation record, and so no assertion
producer, says so with :func:`without_row_attestations`.
"""

from __future__ import annotations

from typing import Any


def without_row_attestations(storage: Any) -> Any:
    """Make a mock ``storage`` one that keeps no row attestations; returns it."""
    storage.attest_strategy_ledger_row = None
    return storage
