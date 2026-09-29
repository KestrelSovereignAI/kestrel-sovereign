"""P1 reproduction (#3306 review): overlapping projection passes retract a row.

Pass A builds its proposal plan. Before A reads the store, pass B -- a separate
task -- adds a NEW ledger row and projects it. A's read now returns B's
assertion, but A's keep-set was frozen before that row existed, so A reconciles
it away. Retraction is terminal for this adapter, so the row is permanently
gone.

The two passes are genuine concurrent tasks, not a re-entrant call: that is the
real shape of the defect, and it is the shape the per-ledger lock is meant to
make impossible. A never awaits B while holding the lock -- it only yields the
event loop -- so the test cannot deadlock when the fix is present. Without the
fix B completes inside that yield and A retracts its row; with the fix B is
held at the lock until A is done, and both rows survive.
"""

from __future__ import annotations

import asyncio

from kestrel_sovereign.knowledge import AssertionStatus
from tests.unit import test_strategic_memory_ledger_assertions as base

# Fixtures are re-exported by binding, not by `from ... import`: a test
# signature that takes `governed`/`ledger` would otherwise read as F811
# redefinition of the imported names.
every = base.every
project = base.project
seed = base.seed
tenant_identity = base.tenant_identity
governed = base.governed
ledger = base.ledger


async def test_overlapping_passes_do_not_retract_a_concurrently_added_row(
    governed, ledger
):
    storage, _raw, _tenant = governed
    seed(ledger)
    await project(storage, ledger)

    original_read = storage._read_ledger_assertions
    state: dict = {"fired": False, "pass_b": None}

    async def racing_read(plan, report):
        # Fires AFTER pass A built its plan and BEFORE pass A reads the store.
        if not state["fired"]:
            state["fired"] = True
            ledger.add_pattern(
                "Added between plan and read", source="#3051", implication="keep me"
            )
            ledger.normalize()
            assert ledger.save() is None
            # Start pass B as an independent task and give the loop a real
            # chance to run it to completion. Unserialized, it finishes here.
            state["pass_b"] = asyncio.create_task(
                project(storage, ledger)
            )
            for _ in range(50):
                if state["pass_b"].done():
                    break
                await asyncio.sleep(0.01)
        return await original_read(plan, report)

    storage._read_ledger_assertions = racing_read
    try:
        await project(storage, ledger)  # pass A
    finally:
        storage._read_ledger_assertions = original_read

    if state["pass_b"] is not None:
        await asyncio.wait_for(state["pass_b"], timeout=30)

    rows = await every(storage)
    retracted = [r for r in rows if r.status is not AssertionStatus.ACTIVE]
    assert state["fired"], "the racing hook never fired - repro is invalid"
    assert not retracted, (
        "pass A retracted a row pass B had just added: "
        f"{[str(r.assertion_id) for r in retracted]}"
    )
