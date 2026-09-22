"""Round-6 reproduction (#3306 / #3320): a failed save authorized a retraction.

The producer used to be handed the live, shared ``StrategyLedger`` and read
its ``data`` inside the projection lock -- but the mutation and the save
happened outside that lock, in the feature. So:

1. a successful ``strategy_add_pattern`` saves and starts reindexing;
2. while its graph projection is still in flight, ``strategy_supersede_pattern``
   retires an existing pattern IN MEMORY and its ``save()`` fails;
3. the add's assertion pass resumes, reads the live ledger, sees the pattern
   superseded, and retracts it.

The file still says the pattern is active. Retraction is terminal for this
adapter, so every later reload reports ``blocked_terminal`` for a row that was
never retired on disk. ``needs_save`` could not catch it: it counts minted ids,
not failed writes.

The fix makes that structural. Mutation, save and the snapshot of what the save
confirmed happen under one feature lock, and the producer only ever sees that
immutable snapshot. A failed save produces no snapshot at all.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.features.strategic_memory.ledger import StrategyLedger
from kestrel_sovereign.knowledge import AssertionStatus
from tests.unit import test_strategic_memory_ledger_assertions as base

# Fixtures are re-exported by binding, not by `from ... import`: a test
# signature that takes `governed`/`wired` would otherwise read as F811
# redefinition of the imported names.
active = base.active
every = base.every
project = base.project
seed = base.seed
tenant_identity = base.tenant_identity
governed = base.governed
ledger = base.ledger
wired = base.wired


def _row_assertion(rows, row_id):
    matches = [r for r in rows if r.object.value.endswith(f":{row_id}")]
    assert len(matches) == 1, f"expected one assertion for {row_id}, got {matches}"
    return matches[0]


class _PausedGraphProjection:
    """Park the FIRST subsequent graph projection until released.

    The graph index runs before the assertion producer in ``_reindex_ledger``,
    so parking it holds a successful mutation's assertion pass open -- the
    window the defect needed.
    """

    def __init__(self, feature):
        self.feature = feature
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._original = feature._project_ledger_graph
        self._calls = 0

    async def _gated(self):
        self._calls += 1
        if self._calls == 1:
            self.entered.set()
            await self.release.wait()
        return await self._original()

    def __enter__(self):
        self.feature._project_ledger_graph = self._gated
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.feature._project_ledger_graph = self._original


async def _add_existing_pattern(feature, storage) -> str:
    result = await feature.strategy_add_pattern(
        pattern="An existing lesson", source="#3051", implication="still holds"
    )
    assert result.data["persisted"] is True
    row_id = result.data["pattern_id"]
    assert _row_assertion(await active(storage), row_id).status is (
        AssertionStatus.ACTIVE
    )
    return row_id


async def _supersede_with_failing_save(feature, row_id):
    ledger = feature._ledger
    ledger.save = lambda: "disk full (injected)"
    try:
        result = await feature.strategy_supersede_pattern(
            pattern_id=row_id, reason="learned better"
        )
    finally:
        del ledger.save
    assert result.data["persisted"] is False, "precondition: the save failed"
    return result


async def _reload_report(storage, path):
    restarted = StrategyLedger(path)
    restarted.load()
    return await storage.project_strategy_ledger_assertions(
        restarted.persisted_snapshot
    )


@pytest.mark.asyncio
async def test_a_failed_supersede_does_not_retract_through_a_paused_projection(
    wired, tmp_path
):
    """Codex's repro, end to end through the real tools and the real store."""
    feature, storage = wired
    existing = await _add_existing_pattern(feature, storage)

    with _PausedGraphProjection(feature) as paused:
        adding = asyncio.create_task(
            feature.strategy_add_pattern(
                pattern="A newly learned lesson", source="#3320", implication="i"
            )
        )
        await asyncio.wait_for(paused.entered.wait(), timeout=5)

        # The add has saved and is parked mid-reindex. Now a supersede whose
        # write fails lands on the same shared ledger. It must not wait for
        # the parked projection: projection runs outside the mutation lock.
        await asyncio.wait_for(
            _supersede_with_failing_save(feature, existing), timeout=5
        )

        paused.release.set()
        added = await asyncio.wait_for(adding, timeout=30)
    assert added.data["persisted"] is True

    held = await every(storage)
    assert _row_assertion(held, existing).status is AssertionStatus.ACTIVE, (
        "an unpersisted supersede reached the projection and retracted a "
        "pattern the canonical file still holds as active"
    )
    assert _row_assertion(held, added.data["pattern_id"]).status is (
        AssertionStatus.ACTIVE
    )

    # The canonical file agrees, and a reload finds nothing terminal to trip on.
    report = await _reload_report(storage, tmp_path / "STRATEGY_LEDGER.yaml")
    assert report.blocked_terminal == 0, report.to_dict()
    assert report.retracted == 0
    assert report.unchanged == 2


@pytest.mark.asyncio
async def test_a_supersede_that_later_persists_still_retracts_on_the_next_pass(
    wired, tmp_path
):
    """The symmetric case: the fix is "only what is on disk", not "never retract".

    The failed supersede stays in memory (the tool said so: a partial result).
    The next successful save carries it to disk, and the pass that follows
    that save is the one that retracts -- not before.
    """
    feature, storage = wired
    existing = await _add_existing_pattern(feature, storage)

    with _PausedGraphProjection(feature) as paused:
        adding = asyncio.create_task(
            feature.strategy_add_pattern(
                pattern="A newly learned lesson", source="#3320", implication="i"
            )
        )
        await asyncio.wait_for(paused.entered.wait(), timeout=5)
        await asyncio.wait_for(
            _supersede_with_failing_save(feature, existing), timeout=5
        )
        paused.release.set()
        await asyncio.wait_for(adding, timeout=30)

    assert _row_assertion(await every(storage), existing).status is (
        AssertionStatus.ACTIVE
    ), "retracted before the supersede was on disk"

    persisted = await feature.strategy_add_pattern(
        pattern="A third lesson", source="#3320", implication="i"
    )
    assert persisted.data["persisted"] is True

    held = await every(storage)
    assert _row_assertion(held, existing).status is not AssertionStatus.ACTIVE, (
        "a supersede that reached the canonical file must retract"
    )
    assert len(await active(storage)) == 2
    report = await _reload_report(storage, tmp_path / "STRATEGY_LEDGER.yaml")
    assert (report.blocked_terminal, report.retracted) == (0, 0)


@pytest.mark.asyncio
async def test_a_successful_supersede_during_a_paused_projection_retracts(
    wired, tmp_path
):
    """A persisted supersede retracts even while an older pass is parked.

    The parked add holds a snapshot taken BEFORE the supersede. When it
    resumes it must not reassert the pattern or undo anything the newer pass
    did; it is refused as superseded by the newer confirmed state.
    """
    feature, storage = wired
    existing = await _add_existing_pattern(feature, storage)

    with _PausedGraphProjection(feature) as paused:
        adding = asyncio.create_task(
            feature.strategy_add_pattern(
                pattern="A newly learned lesson", source="#3320", implication="i"
            )
        )
        await asyncio.wait_for(paused.entered.wait(), timeout=5)
        superseded = await asyncio.wait_for(
            feature.strategy_supersede_pattern(
                pattern_id=existing, reason="learned better"
            ),
            timeout=30,
        )
        assert superseded.data["persisted"] is True
        paused.release.set()
        added = await asyncio.wait_for(adding, timeout=30)

    held = await every(storage)
    assert _row_assertion(held, existing).status is not AssertionStatus.ACTIVE
    assert _row_assertion(held, added.data["pattern_id"]).status is (
        AssertionStatus.ACTIVE
    )
    report = await _reload_report(storage, tmp_path / "STRATEGY_LEDGER.yaml")
    assert (report.blocked_terminal, report.retracted, report.projected) == (0, 0, 0)


@pytest.mark.asyncio
async def test_an_older_snapshot_is_refused_after_a_newer_one_projected(
    governed, ledger
):
    """Snapshots are projected later than they are captured, possibly reordered.

    Reconciling against an older confirmed state after a newer one would
    retract whatever the newer save added -- terminally. The producer refuses
    it instead, and says so.
    """
    storage, _, _ = governed
    seed(ledger)
    older = ledger.persisted_snapshot
    added = ledger.add_pattern("Added by the later save", source="s")
    assert ledger.save() is None
    newer = ledger.persisted_snapshot
    assert newer.persisted_sequence > older.persisted_sequence

    first = await storage.project_strategy_ledger_assertions(newer)
    assert first.projected == 3

    stale = await storage.project_strategy_ledger_assertions(older)

    assert stale.skipped_reason == "ledger_snapshot_superseded"
    assert stale.retracted == 0
    assert _row_assertion(await active(storage), added["id"]).status is (
        AssertionStatus.ACTIVE
    )
    # The newest state itself may be projected again.
    again = await storage.project_strategy_ledger_assertions(newer)
    assert (again.skipped_reason, again.unchanged) == (None, 3)


def test_a_snapshot_cannot_see_a_later_mutation(ledger):
    """The snapshot is a deep copy taken at the save, not a view of the ledger."""
    seed(ledger)
    snapshot = ledger.persisted_snapshot
    ledger.patterns[0]["superseded_at"] = "2026-09-22"
    ledger.add_pattern("in memory only")

    rows = snapshot.data["patterns_learned"]
    assert len(rows) == 1
    assert "superseded_at" not in rows[0]
    with pytest.raises(TypeError):
        snapshot.data["patterns_learned"] = []  # type: ignore[index]


def test_a_failed_save_does_not_advance_the_confirmed_snapshot(ledger):
    seed(ledger)
    before = ledger.persisted_snapshot
    ledger.patterns[0]["superseded_at"] = "2026-09-22"
    ledger.path.unlink()
    ledger.path.mkdir()  # write_text onto a directory fails

    assert ledger.save() is not None
    assert ledger.persisted_snapshot is before
