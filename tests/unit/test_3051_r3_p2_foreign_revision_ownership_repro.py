"""Round-3 P2 reproduction (#3306): ownership read spans historical revisions.

``_ledger_assertion_is_ours`` settles ownership with
``list_assertion_sources(held.assertion_id)``, which is ASSERTION-scoped: its
SQL joins ``semantic_assertion_revisions`` for the whole assertion, so a source
occurrence this adapter wrote for ANY past revision still answers for the
CURRENT one. A foreign writer that revises one of our assertions -- keeping the
marker fields, supplying its own provenance -- is therefore still judged ours,
and the next reconciliation retracts it when the ledger row goes away.
Retraction is terminal, so a row this adapter did not author is destroyed.

The store already exposes the precise read: ``list_assertion_revision_sources``
is revision-scoped and exists for exactly this reason ("a superseded source
cannot leak into a new example"). Ownership must ask about the revision it
actually holds.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from kestrel_sovereign.knowledge import DirectLineage, SourceOccurrence
from tests.unit import test_strategic_memory_ledger_assertions as base

# Fixtures are re-exported by binding, not by `from ... import`: a test
# signature that takes `governed`/`ledger` would otherwise read as F811
# redefinition of the imported names.
active = base.active
every = base.every
seed = base.seed
tenant_identity = base.tenant_identity
governed = base.governed
ledger = base.ledger


@pytest.mark.asyncio
async def test_a_foreign_revision_of_our_assertion_is_not_retracted(governed, ledger):
    """We wrote revision 1. A foreign writer wrote the CURRENT revision.

    Assertion-wide provenance still finds our revision-1 source and answers
    "ours", so reconciliation retracts a revision we did not author.
    """
    storage, _raw, tenant_id = governed
    _pattern, _blocker = seed(ledger)

    # Revision 1 is genuinely ours.
    await storage.project_strategy_ledger_assertions(ledger)
    ours = next(
        a
        for a in await active(storage)
        if a.predicate.value.rsplit("/", 1)[-1] == "strategicPattern"
    )

    # A foreign writer supersedes it, keeping the marker fields but bringing
    # its own provenance. Nothing of ours is bound to THIS revision.
    foreign_source = SourceOccurrence(
        source_occurrence_id="source:some-other-producer:r3p2",
        source_kind="conversation",
        locator="conversation:elsewhere",
        received_at="2026-09-15T00:00:00+00:00",
        content_digest="sha256:" + "2" * 64,
        actor=tenant_id,
        selector="body",
    )
    foreign_revision = replace(
        ours,
        revision_id="foreign-writer-revision-r3p2",
        supersedes_revision_id=ours.revision_id,
        lineage=DirectLineage(
            source_occurrence_ids=(foreign_source.source_occurrence_id,)
        ),
    )
    result = await storage.supersede_assertion(
        ours.revision_id,
        foreign_revision,
        source_occurrences=(foreign_source,),
        operation_id="foreign-writer:r3p2",
    )
    assert getattr(result, "accepted", False), "precondition: foreign revision landed"

    # Remove the row: reconciliation retracts anything it believes is ours.
    ledger.data["patterns_learned"] = []
    ledger.normalize()
    assert ledger.save() is None
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.foreign == 1, (
        "ownership was decided from assertion-wide provenance, so a source we "
        "wrote for an EARLIER revision vouched for a current revision authored "
        f"by someone else (report={report.to_dict()})"
    )
    assert report.retracted == 0
    assert any(
        a.revision_id == foreign_revision.revision_id for a in await active(storage)
    )


@pytest.mark.asyncio
async def test_our_own_current_revision_is_still_ours(governed, ledger):
    """The guard must not make the adapter foreign to its own work."""
    storage, _raw, _tenant = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"] = []
    ledger.normalize()
    assert ledger.save() is None
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.retracted == 1
    assert report.foreign == 0
