"""P2 reproduction (#3306 review): projecting before normalized ids are saved.

``normalize()`` mints ids for hand-written rows that have none. Until
``save()`` lands, those ids exist ONLY in memory -- the next load mints
different ones. Assertion identity derives from the row id, so a projection
taken while ``needs_save`` is true addresses rows by an address that is about
to change: after the reload the old assertions match no current row, and
reconciliation retracts them. Retraction is terminal for this adapter, so those
assertions are permanently dead.

The guard is to refuse the pass while ids are unpersisted (the ledger is
canonical and will be saved by its own caller), reported as its own outcome
rather than as a silent no-op.
"""

from __future__ import annotations

from kestrel_sovereign.features.strategic_memory.ledger import PATTERNS_KEY
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


async def test_projection_refuses_while_normalized_ids_are_unpersisted(
    governed, ledger
):
    storage, _raw, _tenant = governed
    seed(ledger)

    # Hand-written row with no id: normalize mints one, in memory only.
    ledger.data.setdefault(PATTERNS_KEY, []).append(
        {"pattern": "Hand-written row with no id", "source": "#3051"}
    )
    ledger.normalize()
    assert ledger.needs_save, "precondition: normalization minted an unsaved id"

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.skipped_reason == "ledger_ids_unpersisted", (
        "projection ran against ids that exist only in memory; those addresses "
        "change on the next load and the rows written here are retracted "
        f"permanently (skipped_reason={report.skipped_reason!r})"
    )

    # Nothing was written under the doomed addresses.
    rows = await every(storage)
    assert not [r for r in rows if r.status is not AssertionStatus.ACTIVE]


async def test_projection_proceeds_once_ids_are_persisted(governed, ledger):
    storage, _raw, _tenant = governed
    seed(ledger)
    ledger.data.setdefault(PATTERNS_KEY, []).append(
        {"pattern": "Hand-written row with no id", "source": "#3051"}
    )
    ledger.normalize()
    assert ledger.save() is None
    assert not ledger.needs_save

    report = await storage.project_strategy_ledger_assertions(ledger)
    assert report.skipped_reason is None
    assert await every(storage), "persisted ids must project normally"
