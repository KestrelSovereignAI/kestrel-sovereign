"""The strategy-ledger assertion producer, against the REAL governed store (#3051).

``semantic_assertions`` was built, versioned and empty. These tests exist to
prove one producer fills it end to end, and they deliberately use a real
SQLite-backed :class:`AsyncStorage` behind the real
:class:`PrivacyEnforcingStorage` rather than a test double.

That choice is the point, not a preference. A previous attempt at this ticket
passed 43 tests against an in-memory fake and was rejected for five defects the
fake could not express: the fake deleted retracted entries (so a re-add looked
like a fresh write), kept no operation-receipt ledger (so a replayed operation
looked like a successful one), minted no tombstone revision (so a retracted
assertion looked foreign), and had no privacy-transition lease at all. Every
lifecycle assertion below was reproduced against the canonical store first.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path

import pytest

from kestrel_sovereign.features.strategic_memory import ledger_assertions
from kestrel_sovereign.features.strategic_memory.ledger import StrategyLedger
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity
from kestrel_sovereign.knowledge import (
    Assertion,
    AssertionQuery,
    AssertionStatus,
    DirectLineage,
    EpistemicState,
    IRI,
    SourceOccurrence,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
)


# --------------------------------------------------------------------------
# A real loader-verified identity, the same incept -> load boundary boot uses.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tenant_identity(tmp_path_factory):
    identity_dir = tmp_path_factory.mktemp("ledger-assertion-identity")
    credentials = create_kestrel_identity(
        str(identity_dir),
        identity_method="did:pkh",
        agent_name="Ledger Assertion Test",
    )
    tenant_id = credentials.agent_did
    key_id = f"kestrel_{tenant_id.rsplit(':', 1)[-1]}"
    return tenant_id, load_agent_identity(key_id, identity_dir)


@pytest.fixture
async def governed(tenant_identity):
    """A real canonical assertion store behind the real privacy wrapper."""
    tenant_id, identity = tenant_identity
    raw = AsyncStorage(
        ":memory:",
        agent_id=tenant_id,
        _assertion_tenant_capability=(
            _resolve_authenticated_agent_assertion_capability(tenant_id, identity)
        ),
    )
    await raw.initialize()
    try:
        yield PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL), raw, tenant_id
    finally:
        await raw.close()


@pytest.fixture
def ledger(tmp_path: Path) -> StrategyLedger:
    book = StrategyLedger(tmp_path / "STRATEGY_LEDGER.yaml")
    book.load()
    # Persist immediately. Production never projects a ledger it has not
    # written -- ``_persisted_ledger_result`` saves first and only reindexes
    # once the write is confirmed -- and the producer now refuses a ledger
    # whose canonical file was never there, because that is indistinguishable
    # from one whose rows were all deleted.
    assert book.save() is None
    return book


async def active(storage) -> list[Assertion]:
    return await storage.query_assertions(AssertionQuery(limit=500))


async def every(storage) -> list[Assertion]:
    return await storage.query_assertions(
        AssertionQuery(
            statuses=ledger_assertions.statuses_for_current_read(), limit=500
        )
    )


def seed(book: StrategyLedger):
    pattern = book.add_pattern(
        "Reviews find real defects", source="#3051", implication="keep them"
    )
    blocker = book.add_blocker(
        "#42", "Blocked on infra", "high", "me", notes="n", repo="o/r"
    )
    book.normalize()
    assert book.save() is None
    return pattern, blocker


# --------------------------------------------------------------------------
# The mapping: pinned vocabulary, stable identity, no prose in an IRI.
# --------------------------------------------------------------------------


def test_predicates_are_declared_terms_of_the_pinned_ontology():
    """The producer writes against kestrel-vocab 1.2.0, digest and all.

    The ontology is a governed artifact: the version, the content digest and
    the declared term list all travel with the assertion, so a later release
    cannot retroactively reinterpret what was written against this one.
    """
    from kestrel_sovereign.knowledge.registry import get_knowledge_registry

    resource = get_knowledge_registry().resolve("kestrel-vocab", "1.2.0")
    ontology = ledger_assertions.ontology()
    assert ontology.version == "1.2.0"
    assert ontology.content_digest == resource.sha256
    assert ontology.namespace == resource.namespace

    terms = {ledger_assertions.section_term(s) for s in ledger_assertions.sections()}
    assert terms == {"strategicPattern", "strategicBlocker"}
    for term in terms:
        assert f"kestrel:{term}" in resource.selected_terms

    # The TTL the digest is over must actually declare them, or the pin is a
    # promise about bytes that do not contain the terms it names.
    source = Path(resource.package_resource)
    body = (Path(__file__).resolve().parents[2] / "kestrel_sovereign" / source).read_text()
    for term in terms:
        assert f"kestrel:{term} a rdf:Property" in body


def test_the_pinned_ontology_release_is_immutable():
    """1.2.0 adds terms; it does not edit what 1.1.0 already declared."""
    from kestrel_sovereign.knowledge.registry import get_knowledge_registry

    registry = get_knowledge_registry()
    older = registry.resolve("kestrel-vocab", "1.1.0")
    newer = registry.resolve("kestrel-vocab", "1.2.0")
    assert older.sha256 != newer.sha256
    assert set(older.selected_terms) < set(newer.selected_terms)


def test_assertion_identity_is_stable_across_an_edit(tenant_identity, ledger):
    """The object is the row IRI, so identity hashes nothing an edit changes."""
    tenant_id, _ = tenant_identity
    pattern, _ = seed(ledger)

    before = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    ledger.patterns[0]["pattern"] = "Completely different wording"
    after = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)

    by_row = {p.row_id: p for p in before.proposals}
    now = {p.row_id: p for p in after.proposals}
    assert by_row[pattern["id"]].assertion_id == now[pattern["id"]].assertion_id
    # ... and the revision digest is what moved.
    assert by_row[pattern["id"]].content_digest != now[pattern["id"]].content_digest


def test_no_row_prose_reaches_the_canonical_terms(tenant_identity, ledger):
    """Prose stays in the YAML; an IRI is an address, not a claim body."""
    tenant_id, _ = tenant_identity
    seed(ledger)
    plan = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    assert plan.proposals
    for proposal in plan.proposals:
        rendered = " ".join(
            (proposal.subject.value, proposal.predicate.value, proposal.object.value)
        )
        assert "Reviews find real defects" not in rendered
        assert "Blocked on infra" not in rendered


def test_a_row_id_that_is_not_iri_safe_is_refused_not_encoded(tenant_identity):
    """Percent-encoding would collide two distinct hand-written ids onto one."""
    tenant_id, _ = tenant_identity
    plan = ledger_assertions.build_proposal_plan(
        {"patterns_learned": [{"id": "bad id/with slash", "pattern": "p"}]},
        tenant_id=tenant_id,
    )
    assert plan.proposals == ()
    assert plan.unmappable == 1


# --------------------------------------------------------------------------
# Lifecycle, on the canonical store.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_projection_writes_both_row_types(governed, ledger):
    storage, _, tenant_id = governed
    seed(ledger)

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.to_dict() == {
        "projected": 2,
        "revised": 0,
        "unchanged": 0,
        "retracted": 0,
        "skipped": 0,
        "foreign": 0,
        "blocked_terminal": 0,
        "unmappable": 0,
        "failed": 0,
    }
    held = await active(storage)
    assert len(held) == 2
    predicates = {a.predicate.value.rsplit("/", 1)[-1] for a in held}
    assert predicates == {"strategicPattern", "strategicBlocker"}


@pytest.mark.asyncio
async def test_producer_sets_lineage_and_owning_agent_at_write_time(governed, ledger):
    """#3060 needs real lineage on real assertions, not a backfill."""
    storage, _, tenant_id = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    for assertion in await active(storage):
        assert assertion.owning_agent_id == tenant_id
        assert isinstance(assertion.lineage, DirectLineage)
        assert assertion.lineage.source_occurrence_ids
        assert assertion.epistemic_state is EpistemicState.ASSERTED
        sources = await storage.list_assertion_sources(assertion.assertion_id)
        assert ledger_assertions.has_adapter_provenance(sources)


@pytest.mark.asyncio
async def test_reprojection_is_idempotent(governed, ledger):
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert (report.projected, report.revised, report.unchanged) == (0, 0, 2)
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_an_edited_row_revises_rather_than_duplicating(governed, ledger):
    """The property the ticket asked for, measured on the store."""
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)
    before = {a.assertion_id for a in await active(storage)}

    ledger.patterns[0]["implication"] = "a different implication"
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert (report.revised, report.projected) == (1, 0)
    held = await active(storage)
    assert len(held) == 2
    assert {a.assertion_id for a in held} == before
    revised = next(
        a for a in held if a.predicate.value.endswith("strategicPattern")
    )
    assert revised.supersedes_revision_id is not None
    # Lineage survives the edit: the predecessor's own revision is still in
    # history, which is why this is supersession and not delete-then-write.
    revisions = await storage.list_assertion_revisions(revised.assertion_id)
    ours = [
        r for r in revisions
        if ledger_assertions.content_digest_of(r) is not None
    ]
    assert len(ours) == 2
    assert ours[-1].revision_id == revised.revision_id
    assert ledger_assertions.content_digest_of(ours[0]) != (
        ledger_assertions.content_digest_of(ours[-1])
    )


@pytest.mark.asyncio
async def test_a_row_that_oscillates_between_two_states_keeps_projecting(
    governed, ledger
):
    """A -> B -> A -> B. Every hop is a distinct transition of one assertion.

    The oscillation runs twice deliberately. Keying a revision on content alone
    is not caught by a single A->B->A, because the ``write``/``revise`` action
    still separates the first A from the third; it takes a SECOND revise to the
    same content to collide with the immutable revision already stored at that
    id. A revision id is immutable in the canonical store, so the collision
    surfaces as a failed write or a replayed operation reporting a success that
    left the wrong content current.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    for expected in ("B", "keep them", "B", "keep them"):
        ledger.patterns[0]["implication"] = expected
        report = await storage.project_strategy_ledger_assertions(ledger)
        assert (report.revised, report.failed, report.projected) == (1, 0, 0)

        pattern = next(
            a for a in await active(storage)
            if a.predicate.value.endswith("strategicPattern")
        )
        # The CURRENT assertion must carry the row's current content, not a
        # replayed earlier one that merely reported success.
        assert (
            ledger_assertions.content_digest_of(pattern)
            == ledger_assertions.row_content_digest(
                ledger_assertions.PATTERN_SECTION, ledger.patterns[0]
            )
        )

    # And the very next pass agrees nothing is outstanding.
    assert (await storage.project_strategy_ledger_assertions(ledger)).unchanged == 2


@pytest.mark.asyncio
async def test_resolving_a_blocker_retracts_its_assertion(governed, ledger):
    """The live tool retires the row IN PLACE; the claim must stop being held."""
    storage, _, _ = governed
    _, blocker = seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.resolve_blocker(blocker, "fixed upstream")
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.retracted == 1
    held = await active(storage)
    assert [a.predicate.value.rsplit("/", 1)[-1] for a in held] == ["strategicPattern"]
    # The row is still in the canonical file — retirement, not deletion.
    assert len(ledger.blockers) == 1


@pytest.mark.asyncio
async def test_superseding_a_pattern_retracts_its_assertion(governed, ledger):
    storage, _, _ = governed
    pattern, _ = seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.supersede_pattern(pattern, reason="learned better")
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.retracted == 1
    held = await active(storage)
    assert [a.predicate.value.rsplit("/", 1)[-1] for a in held] == ["strategicBlocker"]


@pytest.mark.asyncio
async def test_a_retired_row_stays_retracted_on_the_next_pass(governed, ledger):
    """Reconciliation must not thrash a row it already retracted."""
    storage, _, _ = governed
    _, blocker = seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)
    ledger.resolve_blocker(blocker, "fixed")
    await storage.project_strategy_ledger_assertions(ledger)

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert (report.retracted, report.failed, report.projected) == (0, 0, 0)
    assert report.unchanged == 1


@pytest.mark.asyncio
async def test_a_row_removed_from_the_file_is_retracted(governed, ledger):
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"] = []
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.retracted == 1
    assert len(await active(storage)) == 1


@pytest.mark.asyncio
async def test_re_adding_a_removed_row_reports_blocked_rather_than_lying(
    governed, ledger
):
    """The store replays the original receipt; a 'projected' here would be false.

    This is the defect the in-memory fake hid: it deleted retracted entries and
    kept no receipt ledger, so the replayed write looked like a fresh success
    while the canonical assertion stayed terminal.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)
    ledger.data["patterns_learned"] = []
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.add_pattern(
        "Reviews find real defects", source="#3051", implication="keep them"
    )
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.blocked_terminal == 1
    assert report.projected == 0
    assert report.foreign == 0  # It IS ours; it is merely terminal.
    assert report.needs_attention is True
    assert [a.predicate.value.rsplit("/", 1)[-1] for a in await active(storage)] == [
        "strategicBlocker"
    ]


@pytest.mark.asyncio
async def test_duplicate_row_ids_refuse_the_section_whole(governed, ledger):
    """Two rows at one address: neither written, and nothing reconciled either.

    A keep-set built from an ambiguous section cannot tell "removed" from "lost
    a collision", so the sibling section proceeds and this one does not.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"] = [
        {"id": "dup", "pattern": "one"},
        {"id": "dup", "pattern": "two"},
    ]
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.refused_sections == {"patterns_learned": "duplicate_row_ids"}
    assert (report.projected, report.retracted, report.failed) == (0, 0, 0)
    # The original pattern assertion survives: refusing is not retracting.
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_blanking_a_rows_text_protects_it_rather_than_retracting(
    governed, ledger
):
    """Present-but-silent is not absent, and here the difference is terminal.

    A row whose text is blanked still exists and is still active; it just
    currently asserts nothing. Retracting it would be irreversible — restoring
    the text afterwards reports ``blocked_terminal`` forever — so it is kept out
    of both the proposal set and the reconciliation keep-set's removal side.
    Duplicate row ids get whole-section refusal for the same class of reason.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)
    assert len(await active(storage)) == 2

    ledger.patterns[0]["pattern"] = "   "
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.skipped == 1
    assert report.retracted == 0
    assert len(await active(storage)) == 2

    # Restoring the text re-projects, which is only possible because nothing
    # was retracted.
    ledger.patterns[0]["pattern"] = "Reviews find real defects"
    restored = await storage.project_strategy_ledger_assertions(ledger)
    assert (restored.unchanged, restored.blocked_terminal) == (2, 0)


@pytest.mark.asyncio
async def test_an_unmappable_row_id_refuses_its_section_whole(governed, ledger):
    """An id we cannot compute is an assertion we cannot protect individually.

    So the section is refused entire, exactly as for duplicates — the only
    option that cannot end in retracting a row that is still present.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"].append(
        {"id": "hand/edited/id", "pattern": "still here"}
    )
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.refused_sections == {"patterns_learned": "unmappable_row_ids"}
    assert report.unmappable == 1
    assert report.retracted == 0
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_an_id_less_row_refuses_its_section(governed, ledger):
    """An id-less row is addressed by a digest of its own text.

    So editing that text moves its address, orphaning the assertion written
    under the old one — terminally. ``normalize`` mints ids and the feature
    persists them before projecting, so this is unreachable on the healthy
    path; it becomes reachable once that save has failed. A keep-set must not
    depend on a write having succeeded somewhere else.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"].append({"pattern": "no id on this row"})
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.refused_sections == {"patterns_learned": "unaddressed_rows"}
    assert report.retracted == 0
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_a_malformed_row_refuses_its_section(governed, ledger):
    """A non-mapping member is dropped before either classification sees it.

    It is therefore invisible to the keep-set while its section stays in scope
    for reconciliation, so one mangled YAML line terminally retracts whatever
    that row used to assert.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.data["patterns_learned"].append("a bare string, not a row")
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.refused_sections == {"patterns_learned": "malformed_rows"}
    assert report.retracted == 0
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_the_producer_refuses_a_bare_mapping(governed):
    """A mapping cannot carry readability or file presence, and both gate a
    retraction sweep that cannot be undone."""
    storage, _, _ = governed
    with pytest.raises(TypeError, match="StrategyLedger"):
        await storage.project_strategy_ledger_assertions(
            {"patterns_learned": [], "blockers": []}
        )


@pytest.mark.asyncio
async def test_a_ledger_larger_than_one_read_page_projects_in_one_pass(
    governed, ledger
):
    """No write budget and no read cap: one pass leaves nothing deferred."""
    storage, _, _ = governed
    count = ledger_assertions.READ_PAGE_SIZE + 17
    for index in range(count):
        ledger.add_pattern(f"pattern number {index}", source="bulk")
    ledger.normalize()

    first = await storage.project_strategy_ledger_assertions(ledger)
    assert first.projected == count
    assert len(await active(storage)) == count

    # And the second pass sees every one of them as unchanged, which is only
    # true if the read paged to exhaustion rather than stopping at a cap.
    second = await storage.project_strategy_ledger_assertions(ledger)
    assert second.unchanged == count
    assert (second.projected, second.retracted) == (0, 0)


@pytest.mark.asyncio
async def test_paging_survives_tombstones_filling_whole_pages(governed, ledger):
    """Retired rows keep occupying cursor positions, so the read must page past them.

    The read deliberately includes terminal statuses, so a retracted row is
    still a row the cursor has to step over. Retire enough of them to fill
    whole pages and the failure mode is silent and one-directional: a cursor
    that stopped early would report the rows beyond it as never written, write
    over them, and then retract whatever fell off the far end on the pass after
    that. Asserting the steady state is what catches it — ``unchanged`` must
    account for every surviving row with nothing projected and nothing retracted.
    """
    storage, _, _ = governed
    retired = ledger_assertions.READ_PAGE_SIZE + 20
    surviving = 30
    for index in range(retired + surviving):
        ledger.add_pattern(f"pattern number {index}", source="bulk")
    ledger.normalize()
    assert ledger.save() is None

    first = await storage.project_strategy_ledger_assertions(ledger)
    assert first.projected == retired + surviving

    # Retire in place, exactly as ``strategy_supersede_pattern`` does.
    for row in ledger.data["patterns_learned"][:retired]:
        row["superseded_at"] = "2026-01-01"
    assert ledger.save() is None

    second = await storage.project_strategy_ledger_assertions(ledger)
    assert (second.retracted, second.unchanged) == (retired, surviving)

    # The pass that would expose an early-stopping cursor: every surviving row
    # is still recognized, and no tombstone is rewritten or re-retracted.
    third = await storage.project_strategy_ledger_assertions(ledger)
    assert third.unchanged == surviving
    assert (third.projected, third.revised, third.retracted) == (0, 0, 0)
    assert not third.needs_attention
    assert len(await active(storage)) == surviving

    # And an edit to a row sitting past the first page still revises rather
    # than duplicating, which needs the cursor to have reached it at all.
    ledger.data["patterns_learned"][-1]["implication"] = "changed"
    assert ledger.save() is None
    fourth = await storage.project_strategy_ledger_assertions(ledger)
    assert (fourth.revised, fourth.unchanged) == (1, surviving - 1)
    assert len(await active(storage)) == surviving


# --------------------------------------------------------------------------
# Ownership: marker strings are necessary, never sufficient.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_foreign_assertion_wearing_our_markers_is_not_superseded(
    governed, ledger
):
    """Any writer can set confidence_method; only we can write our provenance."""
    storage, _, tenant_id = governed
    pattern, _ = seed(ledger)
    plan = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    proposal = next(p for p in plan.proposals if p.row_id == pattern["id"])

    foreign_source = SourceOccurrence(
        source_occurrence_id="source:some-other-producer:1",
        source_kind="conversation",
        locator="conversation:elsewhere",
        received_at="2026-09-15T00:00:00+00:00",
        content_digest="sha256:" + "0" * 64,
        actor=tenant_id,
        selector="body",
    )
    binding = storage._acquire_ledger_assertion_binding()
    try:
        impostor = ledger_assertions.build_assertion(
            binding=binding,
            proposal=proposal,
            source=foreign_source,
            transition="f" * 64,
        )
    finally:
        storage._release_ledger_assertion_lease()
    # A real competing writer does not reproduce this adapter's revision
    # grammar, so the cheap unchanged-check cannot fire and ownership is
    # settled the only way it can be: from provenance.
    impostor = replace(impostor, revision_id="foreign-writer-revision-1")
    written = await storage.put_assertion(
        impostor,
        source_occurrences=(foreign_source,),
        operation_id="foreign-writer:1",
    )
    assert written.accepted

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.foreign == 1
    assert (report.revised, report.retracted) == (0, 0)
    survivor = next(
        a for a in await active(storage)
        if a.predicate.value.endswith("strategicPattern")
    )
    assert survivor.revision_id == impostor.revision_id


@pytest.mark.asyncio
async def test_a_foreign_assertion_is_not_retracted_by_reconciliation(
    governed, ledger
):
    """Reconciliation deletes nothing it did not write, same as the graph index."""
    storage, _, tenant_id = governed
    pattern, _ = seed(ledger)
    plan = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    proposal = next(p for p in plan.proposals if p.row_id == pattern["id"])
    foreign_source = SourceOccurrence(
        source_occurrence_id="source:some-other-producer:2",
        source_kind="conversation",
        locator="conversation:elsewhere",
        received_at="2026-09-15T00:00:00+00:00",
        content_digest="sha256:" + "1" * 64,
        actor=tenant_id,
        selector="body",
    )
    binding = storage._acquire_ledger_assertion_binding()
    try:
        impostor = ledger_assertions.build_assertion(
            binding=binding,
            proposal=proposal,
            source=foreign_source,
            transition="e" * 64,
        )
    finally:
        storage._release_ledger_assertion_lease()
    impostor = replace(impostor, revision_id="foreign-writer-revision-2")
    await storage.put_assertion(
        impostor,
        source_occurrences=(foreign_source,),
        operation_id="foreign-writer:2",
    )

    # Now remove the row entirely: reconciliation would retract it if it
    # believed the assertion were ours.
    ledger.data["patterns_learned"] = []
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.foreign == 1
    assert report.retracted == 0
    assert any(
        a.revision_id == impostor.revision_id for a in await active(storage)
    )


@pytest.mark.asyncio
async def test_an_impostor_matching_our_content_exactly_is_still_never_mutated(
    governed, ledger
):
    """The documented limit of the cheap unchanged-check, pinned as a test.

    An assertion reproducing our markers, our revision grammar AND the row's
    exact content digest is counted ``unchanged`` without a provenance read, so
    it is reported as unchanged rather than foreign. What must remain true — and
    is what P1 "markers are not ownership" actually asked for — is that it is
    never superseded and never retracted.
    """
    storage, _, tenant_id = governed
    pattern, _ = seed(ledger)
    plan = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    proposal = next(p for p in plan.proposals if p.row_id == pattern["id"])
    foreign_source = SourceOccurrence(
        source_occurrence_id="source:some-other-producer:4",
        source_kind="conversation",
        locator="conversation:elsewhere",
        received_at="2026-09-15T00:00:00+00:00",
        content_digest="sha256:" + "3" * 64,
        actor=tenant_id,
        selector="body",
    )
    binding = storage._acquire_ledger_assertion_binding()
    try:
        # Built from OUR proposal with no revision-id override, so it carries
        # our exact revision grammar and content digest.
        impostor = ledger_assertions.build_assertion(
            binding=binding,
            proposal=proposal,
            source=foreign_source,
            transition=ledger_assertions.transition_digest(
                content_digest=proposal.content_digest,
                predecessor_revision_id=None,
                action=ledger_assertions.ACTION_WRITE,
            ),
        )
    finally:
        storage._release_ledger_assertion_lease()
    await storage.put_assertion(
        impostor,
        source_occurrences=(foreign_source,),
        operation_id="foreign-writer:4",
    )

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert (report.revised, report.retracted) == (0, 0)
    survivor = next(
        a for a in await active(storage)
        if a.predicate.value.endswith("strategicPattern")
    )
    assert survivor.revision_id == impostor.revision_id


def test_marker_fields_alone_do_not_prove_ownership(tenant_identity, ledger):
    """The unit-level statement of the same rule."""
    tenant_id, _ = tenant_identity
    seed(ledger)
    plan = ledger_assertions.build_proposal_plan(ledger.data, tenant_id=tenant_id)
    proposal = plan.proposals[0]
    foreign = SourceOccurrence(
        source_occurrence_id="source:some-other-producer:3",
        source_kind="conversation",
        locator="conversation:elsewhere",
        received_at="2026-09-15T00:00:00+00:00",
        content_digest="sha256:" + "2" * 64,
        actor=tenant_id,
        selector="body",
    )
    assert ledger_assertions.is_adapter_source(foreign) is False
    assert ledger_assertions.has_adapter_provenance((foreign,)) is False
    ours = ledger_assertions.build_source(
        proposal, transition="d" * 64, owning_agent_id=tenant_id
    )
    assert ledger_assertions.is_adapter_source(ours) is True


# --------------------------------------------------------------------------
# Privacy: the binding and every await share one lease.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_privacy_transition_is_refused_mid_projection(governed, ledger):
    """Otherwise a write commits classified under a policy already replaced."""
    storage, _, _ = governed
    seed(ledger)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = storage.query_assertions

    async def blocked_query(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    storage.query_assertions = blocked_query
    task = asyncio.create_task(storage.project_strategy_ledger_assertions(ledger))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(PrivacyViolationError, match="in flight"):
            storage.set_privacy_mode(PrivacyMode.EPHEMERAL)
        assert storage.privacy_mode is PrivacyMode.NORMAL
    finally:
        release.set()
    report = await asyncio.wait_for(task, timeout=30)
    storage.query_assertions = original

    assert report.projected == 2
    # The completed path released the lease, so the transition now succeeds.
    storage.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert storage.privacy_mode is PrivacyMode.EPHEMERAL


@pytest.mark.asyncio
async def test_a_cancelled_projection_releases_its_lease(governed, ledger):
    storage, _, _ = governed
    seed(ledger)
    entered = asyncio.Event()
    original = storage.query_assertions

    async def hang(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    storage.query_assertions = hang
    task = asyncio.create_task(storage.project_strategy_ledger_assertions(ledger))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    storage.query_assertions = original

    storage.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert storage.privacy_mode is PrivacyMode.EPHEMERAL


@pytest.mark.asyncio
async def test_a_privacy_mode_without_durable_writes_projects_nothing(
    governed, ledger
):
    storage, _, _ = governed
    seed(ledger)
    storage.set_privacy_mode(PrivacyMode.EPHEMERAL)

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.skipped_reason == "privacy_denied"
    assert report.to_dict()["projected"] == 0


# --------------------------------------------------------------------------
# Failure modes that must not become data loss.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreadable_ledger_retracts_nothing(governed, ledger):
    """An unreadable file is not an empty one — that distinction is the fix."""
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    ledger.load_error = "STRATEGY_LEDGER.yaml could not be parsed"
    ledger.data = {"patterns_learned": [], "blockers": []}
    report = await storage.project_strategy_ledger_assertions(ledger)

    assert report.skipped_reason == "ledger_unavailable"
    assert report.retracted == 0
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_a_missing_ledger_file_retracts_nothing(governed, ledger):
    """An unmounted volume must not permanently destroy the projection.

    ``load()`` treats a missing file as a NEW ledger, so ``readable`` stays
    True with zero rows — indistinguishable from "every row was deleted" unless
    the producer asks whether a file was there at all. The graph index tolerates
    the confusion because its writes are upserts; here retraction is terminal,
    so restoring the file afterwards reports ``blocked_terminal`` forever.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)
    assert len(await active(storage)) == 2

    rows = copy.deepcopy(ledger.data)
    ledger.path.unlink()
    # A restart builds a FRESH ledger over the same path. ``load()`` treats the
    # absent file as a NEW ledger, so it holds zero rows while ``readable``
    # stays True -- the exact state that reads as "every row was deleted".
    restarted = StrategyLedger(ledger.path)
    restarted.load()
    assert restarted.readable
    assert restarted.patterns == [] and restarted.blockers == []

    report = await storage.project_strategy_ledger_assertions(restarted)

    assert report.skipped_reason == "ledger_absent"
    assert report.retracted == 0
    assert len(await active(storage)) == 2

    # And the projection is still healthy once the file comes back, which is
    # only true because nothing was retracted: retraction is terminal here.
    restarted.data = rows
    assert restarted.save() is None
    recovered = await storage.project_strategy_ledger_assertions(restarted)
    assert (recovered.unchanged, recovered.blocked_terminal) == (2, 0)
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_a_pathless_ledger_retracts_nothing(governed):
    """``StrategyLedger(None)`` is the same hazard with no file at all."""
    storage, _, _ = governed
    real = StrategyLedger(None)
    real.load()

    report = await storage.project_strategy_ledger_assertions(real)

    assert report.skipped_reason == "ledger_absent"
    assert report.retracted == 0


@pytest.mark.asyncio
async def test_two_byte_identical_rows_both_project(governed, ledger):
    """Two identical rows are two rows, and both must reach the store.

    Every derived identity (revision, source occurrence, operation) descends
    from the row content digest, while the assertion id descends from the row
    id. A content-only digest therefore handed two distinct assertions one
    operation id, and the canonical store correctly refused the second — on
    every reindex, forever. Reachable from the shipped tool: the same pattern
    text twice on the same day, since ``recorded_at`` is date-only.
    """
    storage, _, _ = governed
    first = ledger.add_pattern("Identical wording", source="s", implication="i")
    second = ledger.add_pattern("Identical wording", source="s", implication="i")
    assert first["id"] != second["id"]          # ...but the content matches

    report = await storage.project_strategy_ledger_assertions(ledger)

    assert (report.projected, report.failed) == (2, 0)
    assert len(await active(storage)) == 2

    # The steady state is clean too: the earlier defect surfaced as a permanent
    # per-pass failure rather than a one-off.
    again = await storage.project_strategy_ledger_assertions(ledger)
    assert (again.unchanged, again.failed) == (2, 0)


@pytest.mark.asyncio
async def test_an_unchanged_pass_does_not_read_provenance_per_row(governed, ledger):
    """``revision_id`` carries the digest precisely to avoid this read.

    Reindex runs after every ledger mutation, so a provenance read per held row
    is an await per row per mutation. The steady-state pass must settle
    'unchanged' from the paged query alone.
    """
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    reads = 0
    original = storage.list_assertion_sources

    async def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return await original(*args, **kwargs)

    storage.list_assertion_sources = counted
    try:
        report = await storage.project_strategy_ledger_assertions(ledger)
    finally:
        storage.list_assertion_sources = original

    assert report.unchanged == 2
    assert reads == 0


@pytest.mark.asyncio
async def test_a_failed_assertion_read_retracts_nothing(governed, ledger):
    """'Could not read' authorizes neither a write nor a retraction."""
    storage, _, _ = governed
    seed(ledger)
    await storage.project_strategy_ledger_assertions(ledger)

    async def broken(*args, **kwargs):
        raise RuntimeError("assertion store unavailable")

    original = storage.query_assertions
    storage.query_assertions = broken
    try:
        report = await storage.project_strategy_ledger_assertions(ledger)
    finally:
        storage.query_assertions = original

    assert report.skipped_reason == "assertion_read_failed"
    assert (report.retracted, report.projected) == (0, 0)
    assert len(await active(storage)) == 2


@pytest.mark.asyncio
async def test_the_report_carries_no_row_prose(governed, ledger):
    """Callers log this; a pattern's text must not travel into a log line."""
    storage, _, _ = governed
    seed(ledger)
    report = await storage.project_strategy_ledger_assertions(ledger)

    rendered = repr(report.to_dict())
    assert "Reviews find real defects" not in rendered
    assert "Blocked on infra" not in rendered


# --------------------------------------------------------------------------
# The wiring. A producer nothing calls leaves the store exactly as empty as
# this ticket found it, so these drive the real feature tools end to end.
# --------------------------------------------------------------------------


@pytest.fixture
async def wired(governed, tmp_path):
    """The real StrategicMemoryFeature over the real governed storage."""
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory.feature import (
        StrategicMemoryFeature,
    )

    storage, _, tenant_id = governed
    agent = MagicMock()
    agent.agent_id = tenant_id
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    # The real graph store rides along: the two projections are siblings, and
    # running both here is what shows neither gates the other.

    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature, storage


@pytest.mark.asyncio
async def test_adding_a_pattern_through_the_tool_writes_an_assertion(wired):
    """The end-to-end claim of #3051: a real tool call fills the empty store."""
    feature, storage = wired
    assert await active(storage) == []

    await feature.strategy_add_pattern(
        pattern="Small diffs review better", source="#3051", implication="split work"
    )

    held = await active(storage)
    assert len(held) == 1
    assert held[0].predicate.value.endswith("strategicPattern")
    assert isinstance(held[0].lineage, DirectLineage)


@pytest.mark.asyncio
async def test_adding_a_blocker_through_the_tool_writes_an_assertion(wired):
    feature, storage = wired

    await feature.strategy_add_blocker(
        issue="#77", title="Waiting on review", severity="high", owner="me"
    )

    held = await active(storage)
    assert len(held) == 1
    assert held[0].predicate.value.endswith("strategicBlocker")


@pytest.mark.asyncio
async def test_resolving_a_blocker_through_the_tool_retracts_its_assertion(wired):
    """Through ``strategy_resolve_blocker``, not by deleting a row by hand.

    The live tool retires the row IN PLACE. A projector keyed on the rendered
    claim calls that unchanged and leaves a resolved blocker asserted as
    currently held — which is the substantive defect, and it is only reachable
    through this path.
    """
    feature, storage = wired
    await feature.strategy_add_blocker(
        issue="#77", title="Waiting on review", severity="high", owner="me"
    )
    assert len(await active(storage)) == 1

    await feature.strategy_resolve_blocker(
        issue="#77", resolution="merged"
    )

    assert await active(storage) == []
    terminal = [a for a in await every(storage) if a.status is not AssertionStatus.ACTIVE]
    assert len(terminal) == 1  # retracted, not deleted: lineage survives (#3060)


@pytest.mark.asyncio
async def test_superseding_a_pattern_through_the_tool_retracts_its_assertion(wired):
    feature, storage = wired
    result = await feature.strategy_add_pattern(
        pattern="An old lesson", source="s", implication="i"
    )
    row_id = result.data["pattern_id"]
    assert len(await active(storage)) == 1

    await feature.strategy_supersede_pattern(
        pattern_id=row_id, reason="learned better"
    )

    assert await active(storage) == []


@pytest.mark.asyncio
async def test_editing_the_ledger_and_restarting_reprojects_without_duplicating(
    wired, tmp_path
):
    """A restart re-reads the canonical file; identity must survive it."""
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory.feature import (
        StrategicMemoryFeature,
    )

    feature, storage = wired
    await feature.strategy_add_pattern(
        pattern="Survives a restart", source="s", implication="i"
    )
    before = {a.assertion_id for a in await active(storage)}
    assert len(before) == 1

    # Hand-edit the canonical file the way a human would, then boot again.
    path = tmp_path / "STRATEGY_LEDGER.yaml"
    body = path.read_text(encoding="utf-8")
    path.write_text(body.replace("implication: i", "implication: edited"), "utf-8")

    agent = MagicMock()
    agent.agent_id = feature.agent.agent_id
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    restarted = StrategicMemoryFeature(agent)
    await restarted.initialize()

    held = await active(storage)
    assert {a.assertion_id for a in held} == before  # revised, not duplicated
    assert len(held) == 1


@pytest.mark.asyncio
async def test_the_assertion_producer_does_not_depend_on_the_graph_index(
    governed, tmp_path
):
    """The two projections are siblings, not a chain.

    The graph index derives its node ids from an ``agent_id`` read off the
    agent object and returns early when there is none. The assertion producer
    takes its tenant from the storage binding instead, so it has no business
    inheriting that guard — and sequencing it behind one would reproduce
    exactly the "nothing writes to the store" shape this ticket exists to
    close, silently and only for agents in that state.
    """
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory.feature import (
        StrategicMemoryFeature,
    )

    storage, _, _ = governed
    agent = MagicMock()
    agent.agent_id = ""       # the graph index's early-return condition
    agent.did = ""
    agent.id = ""
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()

    await feature.strategy_add_pattern(
        pattern="Still reaches the assertion store", source="s", implication="i"
    )

    report = await feature._reindex_ledger()
    assert report["skipped_reason"] == "no_agent_identity"   # graph did skip
    assert report["assertions"]["unchanged"] == 1            # ...and this did not
    assert len(await active(storage)) == 1


@pytest.mark.asyncio
async def test_a_failing_producer_never_fails_the_canonical_ledger_write(wired):
    """YAML is canonical and already persisted; a semantic failure is not its failure."""
    feature, storage = wired

    async def broken(*args, **kwargs):
        raise RuntimeError("assertion store exploded")

    storage.project_strategy_ledger_assertions = broken
    result = await feature.strategy_add_pattern(
        pattern="Persisted regardless", source="s", implication="i"
    )

    assert result.data["persisted"] is True
    assert any(
        row["pattern"] == "Persisted regardless" for row in feature._ledger.patterns
    )
