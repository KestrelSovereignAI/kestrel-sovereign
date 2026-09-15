"""The strategy ledger populates ``semantic_assertions`` (#3051).

The store was built, versioned, governed and empty: nothing wrote to it, so
structured agent knowledge had no semantic recall path. These tests pin the
producer that fills it and, more importantly, the four properties the ticket
argued the producer had to have:

* the unit written is the **claim**, one per row;
* identity is the **row**, so an edit is a supersession rather than a second
  competing claim;
* re-rendering is keyed on a **revision digest**, so an unchanged row is a
  no-op and a changed renderer re-projects;
* a row the canonical file dropped is **retracted**, and never retracted from
  a partial or unreadable view of the ledger.

The storage double holds real :class:`Assertion` values, so the identity hash,
the lineage rules, and the supersession constraints are exercised for real
rather than mocked away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    PATTERNS_KEY,
)
from kestrel_sovereign.features.strategic_memory.ledger_assertions import (
    BLOCKER_ASSERTIONS,
    ONTOLOGY_IDENTIFIER,
    ONTOLOGY_VERSION,
    PATTERN_ASSERTIONS,
    STRATEGY_ASSERTION_ADAPTER_VERSION,
    STRATEGY_CLAIM_RENDERER_VERSION,
    claim_revision_digest,
    claim_subject_iri,
    project_ledger_assertions,
    render_blocker_claim,
    render_pattern_claim,
)
from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    Visibility,
    XSD_STRING,
)
from kestrel_sovereign.knowledge.registry import get_knowledge_registry
from kestrel_sovereign.storage.semantic_binding import SemanticAssertionBinding


TENANT = "did:pkh:eip155:1:0xledgertenant"


# --------------------------------------------------------------------------
# A storage double that keeps the governed contract's shape
# --------------------------------------------------------------------------


@dataclass
class _Report:
    report_id: str = "report:test"


@dataclass
class _WriteResult:
    assertion: Optional[Assertion]
    report: _Report

    @property
    def accepted(self) -> bool:
        return self.assertion is not None


@dataclass
class _SupersessionResult:
    predecessor: Optional[Assertion]
    replacement: Optional[Assertion]
    report: _Report

    @property
    def accepted(self) -> bool:
        return self.replacement is not None


class FakeGovernedStorage:
    """A tenant-scoped current-assertion store with the governed write API."""

    def __init__(
        self,
        *,
        binding: Optional[SemanticAssertionBinding] = None,
        binding_error: Optional[Exception] = None,
        reject: bool = False,
    ) -> None:
        self.current: Dict[str, Assertion] = {}
        self.sources: List[Any] = []
        self.retracted: List[str] = []
        self.operations: List[str] = []
        self.binding_operations: List[str] = []
        self._binding = binding or SemanticAssertionBinding(
            tenant_id=TENANT,
            owning_agent_id="agent-kestrel",
            privacy_classification="normal",
            release_policy_reference="policy:privacy:normal-v1",
            visibility=Visibility.PRIVATE,
        )
        self._binding_error = binding_error
        self._reject = reject
        self.page_size_seen: List[int] = []

    def governed_semantic_assertion_binding(
        self, operation: str
    ) -> SemanticAssertionBinding:
        self.binding_operations.append(operation)
        if self._binding_error is not None:
            raise self._binding_error
        return self._binding

    async def query_assertions(self, query=None):
        self.page_size_seen.append(query.limit)
        rows = [
            assertion
            for assertion in self.current.values()
            if query.predicate is None
            or assertion.predicate.value == query.predicate.value
        ]
        rows.sort(key=lambda item: item.revision_id)
        if query.cursor is not None:
            rows = [row for row in rows if row.revision_id > query.cursor]
        return rows[: query.limit]

    async def put_assertion(self, assertion, *, source_occurrences=(), operation_id=None):
        self.operations.append(operation_id)
        self.sources.extend(source_occurrences)
        if self._reject:
            return _WriteResult(assertion=None, report=_Report())
        assert source_occurrences, "a canonical write must carry provenance"
        self.current[assertion.subject.value] = assertion
        return _WriteResult(assertion=assertion, report=_Report())

    async def supersede_assertion(
        self,
        expected_predecessor_revision_id,
        replacement,
        *,
        source_occurrences=(),
        operation_id=None,
    ):
        self.operations.append(operation_id)
        self.sources.extend(source_occurrences)
        predecessor = self.current.get(replacement.subject.value)
        assert predecessor is not None
        assert predecessor.revision_id == expected_predecessor_revision_id
        if self._reject:
            return _SupersessionResult(
                predecessor=None, replacement=None, report=_Report()
            )
        self.current[replacement.subject.value] = replacement
        return _SupersessionResult(
            predecessor=predecessor, replacement=replacement, report=_Report()
        )

    async def retract_assertion(
        self, assertion_id, expected_revision_id, *, operation_id=None
    ):
        self.operations.append(operation_id)
        for subject, assertion in list(self.current.items()):
            if assertion.assertion_id == assertion_id:
                assert assertion.revision_id == expected_revision_id
                self.retracted.append(assertion_id)
                del self.current[subject]
                return object()
        raise AssertionError("retract of an assertion that is not current")


class _Ledger:
    def __init__(self, data: Dict[str, Any], *, readable: bool = True) -> None:
        self.data = data
        self.readable = readable
        self.path = "/tmp/STRATEGY_LEDGER.yaml"


def _ledger(patterns=(), blockers=()) -> _Ledger:
    return _Ledger({PATTERNS_KEY: list(patterns), BLOCKERS_KEY: list(blockers)})


PATTERN_ROW = {
    "id": "pat_stable01",
    "pattern": "A projection that edits its source is no longer derived",
    "implication": "Keep the ledger canonical",
    "recorded_at": "2026-09-01",
}

BLOCKER_ROW = {
    "id": "blk_stable01",
    "title": "Semantic recall has no producer",
    "issue": "#3051",
    "repo": "KestrelSovereignAI/kestrel-sovereign",
    "severity": "high",
}


def _subject(kind: str, row_id: str) -> str:
    return claim_subject_iri(TENANT, kind, row_id)


# --------------------------------------------------------------------------
# The ontology pin
# --------------------------------------------------------------------------


def test_pinned_ontology_declares_both_ledger_predicates() -> None:
    """The predicates are governed terms, not ad-hoc IRI construction.

    ``semantic_facts`` refuses arbitrary predicate-to-IRI construction on the
    grounds that it turns a convenience tool into an ad-hoc ontology authoring
    surface. The same rule binds here, so the terms have to exist in a pinned,
    digest-verified release before anything can be asserted against them.
    """
    registry = get_knowledge_registry()
    resource = registry.resolve(ONTOLOGY_IDENTIFIER, ONTOLOGY_VERSION)
    assert "kestrel:strategicPattern" in resource.selected_terms
    assert "kestrel:strategicBlocker" in resource.selected_terms
    # The digest pin is real: the file on disk must hash to the registry value.
    registry.verify_resources((resource,))


def test_earlier_ontology_release_is_untouched() -> None:
    """1.2.0 is additive; the codec's 1.1 selection must not move."""
    registry = get_knowledge_registry()
    assert (
        registry.select_capability("ontology:kestrel-vocab-1.1").resource.version
        != registry.resolve(ONTOLOGY_IDENTIFIER, ONTOLOGY_VERSION).version
    )


# --------------------------------------------------------------------------
# Rendering: the unit is the claim
# --------------------------------------------------------------------------


def test_pattern_renders_one_claim_including_its_implication() -> None:
    assert render_pattern_claim(PATTERN_ROW) == (
        "A projection that edits its source is no longer derived "
        "Implication: Keep the ledger canonical"
    )


def test_blocker_renders_one_claim_and_keeps_the_repo_with_the_issue() -> None:
    """``#42`` names an issue only relative to a repository."""
    claim = render_blocker_claim(BLOCKER_ROW)
    assert "KestrelSovereignAI/kestrel-sovereign #3051" in claim
    assert claim.startswith("Semantic recall has no producer")


def test_rendering_is_whitespace_canonical() -> None:
    """Re-indenting a YAML block scalar must not mint a new assertion.

    The claim text is the assertion object and therefore part of the identity
    hash, so a purely cosmetic edit that changed it would re-project for no
    semantic reason.
    """
    reindented = dict(PATTERN_ROW)
    reindented["pattern"] = (
        "A projection that edits\n   its source is no longer  derived"
    )
    assert render_pattern_claim(reindented) == render_pattern_claim(PATTERN_ROW)


# --------------------------------------------------------------------------
# Identity: the row, not the text
# --------------------------------------------------------------------------


def test_subject_is_stable_across_a_text_edit() -> None:
    """Identity is keyed on the row id so an edit can be a revision at all.

    If the claim text drove the subject, an edited row would land on a fresh
    subject and the predecessor would be orphaned rather than superseded --
    the duplication the ticket says must not happen.
    """
    edited = dict(PATTERN_ROW, pattern="Rewritten wording entirely")
    assert PATTERN_ASSERTIONS.ledger.row_id(edited) == PATTERN_ASSERTIONS.ledger.row_id(
        PATTERN_ROW
    )
    assert _subject("pattern", PATTERN_ASSERTIONS.ledger.row_id(edited)) == _subject(
        "pattern", PATTERN_ASSERTIONS.ledger.row_id(PATTERN_ROW)
    )


@pytest.mark.parametrize(
    "row_id",
    ["pat_plain", "has space", "slash/and?query", "hash#and%percent", "uni✓code"],
)
def test_subject_iri_survives_normalization_unchanged(row_id: str) -> None:
    """A hand-edited id must stay addressable and stay stable.

    Percent-encoding is injective and ``quote(safe="")`` never escapes an
    unreserved character, so RFC 3986 normalization is a no-op on the result.
    Rejecting odd ids instead would make those rows permanently unreachable.
    """
    value = _subject("pattern", row_id)
    assert IRI(value).value == value


def test_subject_encoding_is_injective_and_leaks_no_delimiter() -> None:
    """Stability is not enough on its own -- the encoding must also be injective.

    Two different rows sharing one subject would make one of them permanently
    unaddressable and let the other's edits supersede it. Decodability is the
    proof: if the final segment unquotes back to the exact row id, no two ids
    can collide. Leaving a reserved delimiter unescaped is how that breaks --
    an unescaped ``#`` turns the rest of the id into a fragment.
    """
    from urllib.parse import unquote

    row_ids = ["a/b", "a%2Fb", "a#b", "a b", "a?b", "a:b", "ab", ""]
    subjects = [_subject("pattern", row_id) for row_id in row_ids]
    assert len(set(subjects)) == len(row_ids)
    for row_id, subject in zip(row_ids, subjects):
        segment = subject.rsplit(":strategy:pattern:", 1)[1]
        assert unquote(segment) == row_id
        assert not set(segment) & set("/?#[]@"), (
            "a reserved delimiter left unescaped stops the segment being the id"
        )


def test_revision_digest_moves_when_the_claim_moves() -> None:
    base = dict(
        tenant_id=TENANT, kind="pattern", row_id="pat_1", claim="one thing"
    )
    assert claim_revision_digest(**base) == claim_revision_digest(**base)
    assert claim_revision_digest(**{**base, "claim": "another"}) != (
        claim_revision_digest(**base)
    )
    assert claim_revision_digest(**{**base, "row_id": "pat_2"}) != (
        claim_revision_digest(**base)
    )
    assert claim_revision_digest(**{**base, "tenant_id": "other"}) != (
        claim_revision_digest(**base)
    )


def test_revision_digest_covers_the_renderer_version(monkeypatch) -> None:
    """A renderer change re-projects rather than leaving stale text asserted.

    The claim text IS the object, so changing how a row renders changes what
    was asserted. A digest that did not cover the renderer version would let
    the old rendering survive under a new contract.
    """
    import kestrel_sovereign.features.strategic_memory.ledger_assertions as module

    before = claim_revision_digest(
        tenant_id=TENANT, kind="pattern", row_id="pat_1", claim="one thing"
    )
    monkeypatch.setattr(module, "STRATEGY_CLAIM_RENDERER_VERSION", "v2")
    after = claim_revision_digest(
        tenant_id=TENANT, kind="pattern", row_id="pat_1", claim="one thing"
    )
    assert before != after
    assert STRATEGY_CLAIM_RENDERER_VERSION == "strategy-ledger-claim-v1"


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_writes_one_assertion_per_row() -> None:
    storage = FakeGovernedStorage()
    report = await project_ledger_assertions(
        storage, _ledger([PATTERN_ROW], [BLOCKER_ROW])
    )
    assert report["projected"] == 2
    assert report["failed"] == 0
    assert set(storage.current) == {
        _subject("pattern", "pat_stable01"),
        _subject("blocker", "blk_stable01"),
    }
    pattern = storage.current[_subject("pattern", "pat_stable01")]
    assert pattern.object.lexical_form == render_pattern_claim(PATTERN_ROW)
    assert pattern.predicate.value.endswith("strategicPattern")


@pytest.mark.asyncio
async def test_written_assertions_declare_lineage_and_owner_at_write_time() -> None:
    """#3060 asked for exactly this, and asked for it at write time.

    The comparison surface that ticket needs only exists once real assertions
    carry a writer-declared lineage for content ``claim_source`` also
    describes. Backfilling later would not produce it.
    """
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    written = storage.current[_subject("pattern", "pat_stable01")]
    assert written.owning_agent_id == "agent-kestrel"
    assert written.tenant_id == TENANT
    assert isinstance(written.lineage, DirectLineage)
    assert written.lineage.source_occurrence_ids == (
        f"source:{STRATEGY_ASSERTION_ADAPTER_VERSION}:"
        + claim_revision_digest(
            tenant_id=TENANT,
            kind="pattern",
            row_id="pat_stable01",
            claim=render_pattern_claim(PATTERN_ROW),
        ),
    )
    assert written.epistemic_state is EpistemicState.REPORTED
    assert written.confidence_method == STRATEGY_ASSERTION_ADAPTER_VERSION


@pytest.mark.asyncio
async def test_reprojecting_unchanged_rows_writes_nothing() -> None:
    """Re-projection upserts; it must not duplicate or churn."""
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW], [BLOCKER_ROW]))
    storage.operations.clear()

    report = await project_ledger_assertions(
        storage, _ledger([PATTERN_ROW], [BLOCKER_ROW])
    )
    assert report["unchanged"] == 2
    assert report["projected"] == 0
    assert report["revised"] == 0
    assert storage.operations == []
    assert len(storage.current) == 2


@pytest.mark.asyncio
async def test_an_edited_row_supersedes_rather_than_duplicating() -> None:
    """The ticket's central requirement: re-project, do not duplicate."""
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    original = storage.current[_subject("pattern", "pat_stable01")]

    edited = dict(PATTERN_ROW, implication="Keep the ledger canonical, always")
    report = await project_ledger_assertions(storage, _ledger([edited]))

    assert report["revised"] == 1
    assert report["projected"] == 0
    assert len(storage.current) == 1, "an edit must not leave a second claim"
    replacement = storage.current[_subject("pattern", "pat_stable01")]
    assert replacement.supersedes_revision_id == original.revision_id
    assert replacement.revision_id != original.revision_id
    assert replacement.assertion_id != original.assertion_id, (
        "identity hashes the object, so a changed claim is a changed id -- "
        "which is why the write has to be a supersession"
    )
    assert replacement.object.lexical_form == render_pattern_claim(edited)


@pytest.mark.asyncio
async def test_a_removed_row_is_retracted() -> None:
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW], [BLOCKER_ROW]))
    retained = storage.current[_subject("blocker", "blk_stable01")].assertion_id

    report = await project_ledger_assertions(storage, _ledger([], [BLOCKER_ROW]))

    assert report["retracted"] == 1
    assert list(storage.current) == [_subject("blocker", "blk_stable01")]
    assert storage.current[_subject("blocker", "blk_stable01")].assertion_id == retained


@pytest.mark.asyncio
async def test_a_text_less_row_is_skipped_not_asserted() -> None:
    storage = FakeGovernedStorage()
    report = await project_ledger_assertions(
        storage, _ledger([{"id": "pat_empty", "pattern": "   "}])
    )
    assert report["skipped"] == 1
    assert report["projected"] == 0
    assert storage.current == {}


# --------------------------------------------------------------------------
# The guards that keep a derived index from destroying canonical content
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_title_less_blocker_is_skipped_even_though_it_would_render() -> None:
    """The renderer alone cannot decide what is projectable.

    A blocker with no title still renders non-empty text out of its severity
    and issue -- "Blocked on X. Severity: high." asserts nothing. Projectability
    is the ledger index's rule about the row's own text, and it has to be the
    gate, because the rendered string is not evidence that there is a claim.
    """
    storage = FakeGovernedStorage()
    report = await project_ledger_assertions(
        storage,
        _ledger([], [{"id": "blk_untitled", "title": "", "issue": "#1",
                      "severity": "high"}]),
    )
    assert report["skipped"] == 1
    assert report["projected"] == 0
    assert storage.current == {}


@pytest.mark.asyncio
async def test_an_unreadable_ledger_retracts_nothing() -> None:
    """An unreadable ledger is not an empty one.

    Reconciliation derives its keep-set from the rows it is given, so treating
    a failed parse as "no rows" reads as "every row was deleted" -- a parse
    error escalated into data loss. The same failure #2851 hit.
    """
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    before = dict(storage.current)

    report = await project_ledger_assertions(
        storage, _Ledger({PATTERNS_KEY: []}, readable=False)
    )

    assert report["skipped_reason"] == "ledger_unavailable"
    assert storage.retracted == []
    assert storage.current == before


@pytest.mark.asyncio
async def test_a_saturated_read_reports_an_unrun_reconcile(monkeypatch) -> None:
    """Never retract against a partial view of what exists."""
    import kestrel_sovereign.features.strategic_memory.ledger_assertions as module

    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    monkeypatch.setattr(module, "_READ_PAGE_SIZE", 1)
    monkeypatch.setattr(module, "_MAX_READ_PAGES", 1)

    report = await project_ledger_assertions(storage, _ledger([]))

    assert report["read_saturated"] is True
    assert report["reconcile_skipped"] == "read_saturated"
    assert storage.retracted == []


@pytest.mark.asyncio
async def test_a_foreign_claim_at_the_same_subject_is_left_alone() -> None:
    """A canonical record this producer did not write is not ours to edit."""
    storage = FakeGovernedStorage()
    subject = _subject("pattern", "pat_stable01")
    foreign = Assertion(
        tenant_id=TENANT,
        owning_agent_id="agent-kestrel",
        subject=IRI(subject),
        predicate=IRI(f"https://kestrel.ai/vocab/strategicPattern"),
        object=Literal("a claim from somewhere else", XSD_STRING),
        revision_id="revision:foreign:1",
        confidence="1",
        confidence_method="some-other-writer",
        confidence_basis="some-other-basis",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-09-01T00:00:00Z",
        ontology_version=OntologyRef(
            "https://kestrel.ai/vocab/", "1.2.0", "sha256:x", "semantic-kb-v1"
        ),
        lineage=DirectLineage(("source:foreign:1",)),
        privacy_classification="normal",
        release_policy_reference="policy:privacy:normal-v1",
    )
    storage.current[subject] = foreign

    report = await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))

    assert report["foreign"] == 1
    assert report["projected"] == 0
    assert storage.current[subject] is foreign


@pytest.mark.asyncio
async def test_reconcile_does_not_retract_a_foreign_claim() -> None:
    storage = FakeGovernedStorage()
    subject = _subject("pattern", "pat_gone")
    storage.current[subject] = Assertion(
        tenant_id=TENANT,
        owning_agent_id="agent-kestrel",
        subject=IRI(subject),
        predicate=IRI("https://kestrel.ai/vocab/strategicPattern"),
        object=Literal("not ours", XSD_STRING),
        revision_id="revision:foreign:2",
        confidence="1",
        confidence_method="some-other-writer",
        confidence_basis="some-other-basis",
        epistemic_state=EpistemicState.REPORTED,
        asserted_at="2026-09-01T00:00:00Z",
        ontology_version=OntologyRef(
            "https://kestrel.ai/vocab/", "1.2.0", "sha256:x", "semantic-kb-v1"
        ),
        lineage=DirectLineage(("source:foreign:2",)),
        privacy_classification="normal",
        release_policy_reference="policy:privacy:normal-v1",
    )

    report = await project_ledger_assertions(storage, _ledger([]))

    assert report["retracted"] == 0
    assert storage.retracted == []
    assert subject in storage.current


@pytest.mark.asyncio
async def test_a_privacy_refusal_is_a_skip_not_a_downgrade() -> None:
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyViolationError

    storage = FakeGovernedStorage(
        binding_error=PrivacyViolationError("blocked in this mode")
    )
    report = await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    assert report["skipped_reason"] == "privacy_denied"
    assert report["projected"] == 0
    assert storage.current == {}


@pytest.mark.asyncio
async def test_an_unwrapped_storage_is_refused() -> None:
    """Raw ``AsyncStorage`` reports ``normal``/PRIVATE unconditionally.

    It cannot see the live privacy policy, so taking tenant/owner/privacy from
    it would silently assert a public-mode claim as private -- or worse. A
    producer that cannot obtain the governed binding does not write.
    """

    class _Unwrapped:
        async def query_assertions(self, query=None):  # pragma: no cover
            raise AssertionError("must not read without a governed binding")

    report = await project_ledger_assertions(_Unwrapped(), _ledger([PATTERN_ROW]))
    assert report["skipped_reason"] == "no_governed_binding"


@pytest.mark.asyncio
async def test_no_storage_is_a_skip() -> None:
    report = await project_ledger_assertions(None, _ledger([PATTERN_ROW]))
    assert report["skipped_reason"] == "no_storage"
    assert report["skipped"] == 1


@pytest.mark.asyncio
async def test_a_rejected_governed_write_is_counted_not_swallowed() -> None:
    """A rejected write leaves the claim absent, so it is not a projection."""
    storage = FakeGovernedStorage(reject=True)
    report = await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    assert report["rejected"] == 1
    assert report["projected"] == 0
    assert storage.current == {}


@pytest.mark.asyncio
async def test_the_write_budget_defers_rather_than_dropping(monkeypatch) -> None:
    """A bounded pass must make monotonic progress, not give up on the rest."""
    import kestrel_sovereign.features.strategic_memory.ledger_assertions as module

    monkeypatch.setattr(module, "_MAX_WRITES_PER_PASS", 1)
    rows = [
        dict(PATTERN_ROW, id=f"pat_{index}", pattern=f"pattern number {index}")
        for index in range(3)
    ]
    storage = FakeGovernedStorage()

    first = await project_ledger_assertions(storage, _ledger(rows))
    assert first["projected"] == 1
    assert first["deferred"] == 2
    assert first["budget_exhausted"] is True

    second = await project_ledger_assertions(storage, _ledger(rows))
    assert second["projected"] == 1
    assert second["unchanged"] == 1, "the finished row is not redone"


@pytest.mark.asyncio
async def test_a_store_failure_never_raises_into_the_ledger_write() -> None:
    """The canonical record is already on disk; an index cannot fail it."""

    class _Broken(FakeGovernedStorage):
        async def query_assertions(self, query=None):
            raise RuntimeError("assertion store unavailable")

    report = await project_ledger_assertions(_Broken(), _ledger([PATTERN_ROW]))
    assert report["failed"] >= 1
    assert report["section_errors"] == [
        PATTERN_ASSERTIONS.noun,
        BLOCKER_ASSERTIONS.noun,
    ]


@pytest.mark.asyncio
async def test_operation_ids_are_content_keyed_for_idempotent_replay() -> None:
    """Two passes over identical content must name the same operation.

    A projector is not a user action: re-running it with the same content is
    the same operation, so the id is derived from the content rather than from
    an invocation.
    """
    storage = FakeGovernedStorage()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    first = list(storage.operations)
    storage.current.clear()
    storage.operations.clear()
    await project_ledger_assertions(storage, _ledger([PATTERN_ROW]))
    assert storage.operations == first
    assert first[0].startswith(f"{STRATEGY_ASSERTION_ADAPTER_VERSION}:put:")


# --------------------------------------------------------------------------
# The wiring: a producer nothing calls is not a producer
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reindex_ledger_runs_the_assertion_producer(tmp_path) -> None:
    """``_reindex_ledger`` must actually reach the assertion store.

    The graph index and the assertion producer are two layers of the same
    reindex, and the failure mode this guards is the one #2851 records: a
    projection that was written, tested in isolation, and never invoked from
    the live path at all.
    """
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature

    storage = FakeGovernedStorage()
    agent = MagicMock()
    agent.agent_id = "agent-kestrel"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    storage.graph = None

    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    feature._ledger = _ledger([PATTERN_ROW], [BLOCKER_ROW])

    report = await feature._reindex_ledger()

    assert report["assertions"]["projected"] == 2
    assert len(storage.current) == 2
    # ``initialize`` reindexes too, so the producer has already run once by
    # here -- which is itself the migration path for an existing ledger.
    assert set(storage.binding_operations) == {"strategy_ledger_assertion"}


@pytest.mark.asyncio
async def test_reindex_ledger_reports_the_two_layers_separately(tmp_path) -> None:
    """A count has to say which layer produced it."""
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature

    agent = MagicMock()
    agent.agent_id = "agent-kestrel"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = FakeGovernedStorage()
    agent.storage.graph = None

    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    feature._ledger = _ledger([PATTERN_ROW])

    report = await feature._reindex_ledger()

    assert report["skipped_reason"] == "no_graph_store", "graph layer reports itself"
    assert report["assertions"]["projected"] == 1, "assertion layer reports itself"


# --------------------------------------------------------------------------
# The governed binding this producer depends on
# --------------------------------------------------------------------------


def _wrapper(mode):
    """A real privacy wrapper over a raw storage that reports a raw binding."""
    from unittest.mock import Mock

    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    raw = Mock()
    raw.semantic_assertion_binding = Mock(
        return_value=SemanticAssertionBinding(
            tenant_id=TENANT,
            owning_agent_id="agent-kestrel",
            privacy_classification="normal",
            release_policy_reference="policy:privacy:normal-v1",
            visibility=Visibility.PRIVATE,
        )
    )
    return PrivacyEnforcingStorage(raw, mode)


def test_governed_binding_keeps_normal_mode_private() -> None:
    from kestrel_sovereign.privacy import PrivacyMode

    binding = _wrapper(PrivacyMode.NORMAL).governed_semantic_assertion_binding(
        "strategy_ledger_assertion"
    )
    assert binding.tenant_id == TENANT
    assert binding.owning_agent_id == "agent-kestrel"
    assert binding.privacy_classification == "normal"
    assert binding.visibility is Visibility.PRIVATE
    assert binding.release_policy_reference == "policy:privacy:normal-v1"


def test_governed_binding_follows_public_sharing() -> None:
    """The raw binding always says ``normal``; the live policy is what decides.

    This is the whole reason a producer must not read
    ``AsyncStorage.semantic_assertion_binding`` directly -- doing so would
    label a public-mode claim private and quietly mis-classify it.
    """
    from kestrel_sovereign.privacy import PrivacyMode

    binding = _wrapper(PrivacyMode.PUBLIC).governed_semantic_assertion_binding(
        "strategy_ledger_assertion"
    )
    assert binding.privacy_classification == "public"
    assert binding.visibility is Visibility.PUBLIC
    assert binding.release_policy_reference == "policy:privacy:public-v1"


@pytest.mark.parametrize(
    "mode_name",
    ["ISOLATED", "EPHEMERAL", "ANONYMOUS", "DEIDENTIFIED"],
)
def test_governed_binding_refuses_modes_without_durable_semantic_authority(
    mode_name: str,
) -> None:
    """Every refusing mode refuses, and names the operation that was refused."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyViolationError

    wrapper = _wrapper(getattr(PrivacyMode, mode_name))
    with pytest.raises(PrivacyViolationError) as excinfo:
        wrapper.governed_semantic_assertion_binding("strategy_ledger_assertion")
    assert "strategy_ledger_assertion" in str(excinfo.value)


@pytest.mark.asyncio
async def test_save_fact_uses_the_same_governed_binding() -> None:
    """``save_fact`` and this producer must share one classification rule.

    The refactor that gave the producer a public accessor only holds if the
    explicit-teaching path really routes through it; a re-inlined private copy
    would drift from this one silently.
    """
    from kestrel_sovereign.agent.invocation import ensure_invocation_id
    from kestrel_sovereign.privacy import PrivacyMode

    wrapper = _wrapper(PrivacyMode.NORMAL)
    seen: List[str] = []

    def _sentinel(operation: str):
        seen.append(operation)
        raise RuntimeError("sentinel-binding")

    wrapper.governed_semantic_assertion_binding = _sentinel

    with pytest.raises(RuntimeError, match="sentinel-binding"):
        await wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="eu-west-1",
            confidence=1.0,
            invocation_id=ensure_invocation_id(None),
        )
    assert seen == ["explicit_fact"]


@pytest.mark.asyncio
async def test_provenance_locator_does_not_leak_the_host_path(tmp_path) -> None:
    """A source locator is durable and travels in an identity export.

    The absolute path is the wrong identifier twice: it discloses where the
    operator keeps agent data, and it is not stable across hosts the same
    agent runs on.
    """
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature

    storage = FakeGovernedStorage()
    agent = MagicMock()
    agent.agent_id = "agent-kestrel"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = storage
    storage.graph = None

    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    feature._ledger = _Ledger(
        {PATTERNS_KEY: [PATTERN_ROW], BLOCKERS_KEY: []},
    )
    feature._ledger.path = f"{tmp_path}/STRATEGY_LEDGER.yaml"

    await feature._reindex_ledger()

    assert storage.sources, "the write must have carried a source occurrence"
    locator = storage.sources[0].locator
    assert str(tmp_path) not in locator
    assert not locator.startswith("/")
    assert locator.startswith("STRATEGY_LEDGER.yaml#")
    # Still followable: the section and the row it came from are named.
    assert locator.endswith(f"{PATTERNS_KEY}:pat_stable01")

    assert feature._ledger_locator() == "STRATEGY_LEDGER.yaml"
    assert not feature._ledger_locator().startswith("/")


def test_ledger_locator_falls_back_when_there_is_no_path(tmp_path) -> None:
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature

    agent = MagicMock()
    agent.agent_data_dir = str(tmp_path)
    feature = StrategicMemoryFeature(agent)
    feature._ledger = _Ledger({})
    feature._ledger.path = None
    assert feature._ledger_locator() == "strategy-ledger"
