"""P2 reproduction (#3306 review): projecting before normalized ids are saved.

``normalize()`` mints ids for hand-written rows that have none. Until
``save()`` lands, those ids exist ONLY in memory -- the next load mints
different ones. Assertion identity derives from the row id, so a projection
taken while those ids are unpersisted addresses rows by an address that is
about to change: after the reload the old assertions match no current row, and
reconciliation retracts them. Retraction is terminal for this adapter, so those
assertions are permanently dead.

Round 6 made this structural rather than guarded (#3320): the producer only
accepts a :class:`LedgerSnapshot`, and only the ledger can confirm one -- after
a save succeeds, or on a load that minted nothing. An in-memory mint is simply
not in any confirmed snapshot. A snapshot the ledger did not vouch for is
refused with its own reason code rather than projected.
"""

from __future__ import annotations

from types import MappingProxyType

from kestrel_sovereign.features.strategic_memory.ledger import (
    PATTERNS_KEY,
    LedgerSnapshot,
    StrategyLedger,
)
from kestrel_sovereign.knowledge import AssertionStatus
from tests.unit import test_strategic_memory_ledger_assertions as base

# Fixtures are re-exported by binding, not by `from ... import`: a test
# signature that takes `governed`/`ledger` would otherwise read as F811
# redefinition of the imported names.
every = base.every
seed = base.seed
tenant_identity = base.tenant_identity
governed = base.governed
ledger = base.ledger


def _mint_unsaved_row(book: StrategyLedger) -> str:
    book.data.setdefault(PATTERNS_KEY, []).append(
        {"pattern": "Hand-written row with no id", "source": "#3051"}
    )
    book.normalize()
    assert book.needs_save, "precondition: normalization minted an unsaved id"
    return book.patterns[-1]["id"]


async def test_an_in_memory_mint_is_not_in_the_confirmed_snapshot(
    governed, ledger
):
    """The confirmed snapshot is what the file holds; the minted row is not."""
    storage, _raw, _tenant = governed
    seed(ledger)
    minted = _mint_unsaved_row(ledger)

    snapshot = ledger.persisted_snapshot
    assert snapshot.confirmed_persisted
    assert minted not in {row.get("id") for row in snapshot.data[PATTERNS_KEY]}

    report = await storage.project_strategy_ledger_assertions(snapshot)

    assert report.skipped_reason is None
    assert report.projected == 2  # the seeded rows only
    rows = await every(storage)
    assert not any(minted in r.object.value for r in rows), (
        "a row id that exists only in memory reached the canonical store"
    )


async def test_a_load_that_minted_ids_is_refused_as_unpersisted(governed, ledger):
    """A reload whose normalization minted ids is not a confirmed state."""
    storage, _raw, _tenant = governed
    seed(ledger)
    _mint_unsaved_row(ledger)
    # Write the id-less row to disk the way a human hand-edit would, bypassing
    # normalization, so the NEXT load has to mint.
    ledger.patterns[-1].pop("id")
    import yaml

    ledger.path.write_text(yaml.safe_dump(dict(ledger.data)), encoding="utf-8")

    restarted = StrategyLedger(ledger.path)
    restarted.load()
    assert restarted.needs_save

    report = await storage.project_strategy_ledger_assertions(
        restarted.persisted_snapshot
    )

    assert report.skipped_reason == "ledger_ids_unpersisted", (
        "projection ran against ids that exist only in memory; those addresses "
        "change on the next load and the rows written here are retracted "
        f"permanently (skipped_reason={report.skipped_reason!r})"
    )
    rows = await every(storage)
    assert rows == []
    assert not [r for r in rows if r.status is not AssertionStatus.ACTIVE]


async def test_a_snapshot_the_ledger_did_not_vouch_for_is_refused(governed, ledger):
    """Constructing a snapshot outside the ledger cannot confirm it.

    Not even by claiming a sequence number: only the ledger module holds the
    witness that makes a snapshot confirmed.
    """
    storage, _raw, _tenant = governed
    seed(ledger)
    forged = LedgerSnapshot(
        path=ledger.path,
        data=MappingProxyType(dict(ledger.data)),
        readable=True,
        has_canonical_file=True,
        persisted_sequence=10**9,
    )
    assert not forged.confirmed_persisted

    report = await storage.project_strategy_ledger_assertions(forged)

    assert report.skipped_reason == "ledger_ids_unpersisted"
    assert await every(storage) == []


async def test_projection_proceeds_once_ids_are_persisted(governed, ledger):
    storage, _raw, _tenant = governed
    seed(ledger)
    minted = _mint_unsaved_row(ledger)
    assert ledger.save() is None
    assert not ledger.needs_save

    report = await storage.project_strategy_ledger_assertions(
        ledger.persisted_snapshot
    )
    assert report.skipped_reason is None
    assert report.projected == 3
    assert any(minted in r.object.value for r in await every(storage))
