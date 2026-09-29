"""Map strategy-ledger rows onto canonical semantic assertions (#3051).

``semantic_assertions`` was built, versioned and governed, and held zero rows:
the substrate for semantic recall existed with no producer. This module is the
first one — deliberately the narrowest, so that "an assertion store with no
producer" is discharged by one row type working end to end rather than by three
multiplying the vocabulary and the reconciliation surface at once.

Like :mod:`semantic_facts`, this module owns a **bounded, versioned mapping**
and nothing else. It builds proposals; it never reaches through a storage
facade, never writes, and never decides privacy. The awaits live in
``PrivacyEnforcingStorage``, which holds a privacy-transition lease across all
of them (see :meth:`PrivacyEnforcingStorage.project_strategy_ledger_assertions`).

The shape of the claim
----------------------

``STRATEGY_LEDGER.yaml`` stays canonical, exactly as :mod:`ledger_index`
established for the graph. The assertion is derived, and its terms are::

    subject   urn:kestrel:agent:<tenant>:strategy-ledger
    predicate kestrel:strategicPattern | kestrel:strategicBlocker
    object    urn:kestrel:agent:<tenant>:ledgerRow:<row_id>

The row TYPE is the predicate and the STABLE ROW IDENTITY is the object.  Two
consequences, both load-bearing:

1. :func:`derive_assertion_id` hashes ``tenant + subject + predicate + object``,
   so it hashes nothing an edit changes. Editing a pattern's wording writes a
   **new revision of the same assertion** rather than minting a second one —
   "re-projects rather than duplicating", which is the property the ticket
   asked for and which row prose in the object would have destroyed.
2. No prose enters an IRI. ``semantic_facts`` refuses ad-hoc predicate
   construction precisely so a convenience tool cannot become an ontology
   authoring surface; encoding the row type (or its text) in an opaque string
   would smuggle the same thing back in. The two predicates are declared terms
   of the immutable ``kestrel-vocab`` 1.2.0 pin and are resolved through the
   registry, digest and all.

Prose therefore does not travel into semantic recall with these assertions, and
that is intended rather than a residue: ``_claim_text`` renders
``subject | predicate | object``, and teaching it to join the strategy ledger
would couple a generic renderer to one feature. Keyword search over the rows'
own text is already served by ``strategy_search`` and is unaffected here.

Lifecycle
---------

One rule: **an adapter assertion is active exactly while its row is present,
projectable and active in the canonical ledger.**

======================  ===========================================
canonical ledger        canonical assertion
======================  ===========================================
row added               initial governed write
row edited              supersession; predecessor becomes superseded
row retired in place    retracted (``superseded_at``/``resolved_at``)
row removed from YAML   retracted
======================  ===========================================

Retirement retracts because ``strategy_supersede_pattern`` and
``strategy_resolve_blocker`` retire a row *in place* — the row keeps existing
and only its lifecycle field changes. A projector that keyed on the rendered
claim alone would call that "unchanged" and leave a resolved blocker asserted
as current, which is the substantive defect: recall would serve retired
knowledge as held.

Retraction is terminal for this adapter. The canonical store offers
reactivation only to inferred assertions and to the ``save_fact`` adapter's own
deleted shells, so a row whose id becomes active again after retraction cannot
be revived here. The shipped tools never produce that input — ``add_pattern``
mints a fresh suffixed id when a retired row already holds one — so it is
reachable only by hand-editing the YAML. When it happens the projector reports
``blocked_terminal`` and writes nothing. It does **not** call ``put_assertion``
and report success: the store would replay the original accepted receipt and
the assertion would stay retracted, which is a write that lies about its own
outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from kestrel_sovereign.agent.invocation import current_invocation_provenance
from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    IRI,
    OntologyRef,
    SourceOccurrence,
)
from kestrel_sovereign.knowledge.registry import get_knowledge_registry
from kestrel_sovereign.storage.semantic_binding import SemanticAssertionBinding

from .ledger import BLOCKERS_KEY, LEDGER_FILENAME, PATTERNS_KEY
from .ledger_index import BLOCKER_SECTION, PATTERN_SECTION, LedgerSection

#: The closed local-term, identity and provenance grammar owned by this
#: adapter. Bumping it is a migration, not an edit: every identity below is
#: derived from it, so a changed value re-keys revisions, sources and
#: operations and stops recognizing what the previous version wrote.
LEDGER_ADAPTER_VERSION = "strategic-memory-ledger-v1"

ONTOLOGY_IDENTIFIER = "kestrel-vocab"
ONTOLOGY_VERSION = "1.2.0"

_CONFIDENCE_METHOD = LEDGER_ADAPTER_VERSION
_CONFIDENCE_BASIS = "canonical-strategy-ledger-row"
_SOURCE_KIND = "agent_strategy_ledger"
_SELECTOR = "ledger-row"
_LOCATOR_STEM = f"ledger:{LEDGER_FILENAME}#"

_REVISION_PREFIX = f"{LEDGER_ADAPTER_VERSION}:r1:"
_SOURCE_PREFIX = f"source:{LEDGER_ADAPTER_VERSION}:"
_SUBJECT_SUFFIX = "strategy-ledger"
_OBJECT_INFIX = "ledgerRow"

#: The predicate term and the exact row fields whose content the revision
#: represents. An explicit list rather than "every string field": a field added
#: to the YAML later should be a deliberate re-projection, not a silent one.
_PATTERN_TERM = "strategicPattern"
_BLOCKER_TERM = "strategicBlocker"
_SECTION_TERMS = {PATTERNS_KEY: _PATTERN_TERM, BLOCKERS_KEY: _BLOCKER_TERM}
_SECTION_CONTENT_FIELDS = {
    PATTERNS_KEY: (
        "pattern",
        "source",
        "implication",
        "recorded_at",
        "superseded_at",
        "superseded_by",
        "superseded_reason",
    ),
    BLOCKERS_KEY: (
        "issue",
        "title",
        "severity",
        "owner",
        "repo",
        "notes",
        "blocked_since",
        "resolved_at",
        "resolution",
    ),
}

#: Row ids reach IRIs, and the ledger is hand-editable, so the charset is the
#: RFC 3986 unreserved set and nothing else. A row whose id falls outside it is
#: reported rather than percent-encoded: an encoding would make two different
#: hand-written ids collide onto one assertion.
_ROW_ID_RE = re.compile(r"^[A-Za-z0-9._~-]{1,128}$")

_SECTIONS = (PATTERN_SECTION, BLOCKER_SECTION)

#: Every terminal action this adapter can take, for the transition digest.
ACTION_WRITE = "write"
ACTION_REVISE = "revise"
ACTION_RETRACT = "retract"

#: Page size for the executor's read of current adapter assertions.
#:
#: The read is deliberately paged to exhaustion rather than capped: it is the
#: keep-set's own baseline, and a saturated page is indistinguishable from a
#: complete one. Capping it would make every assertion past the cap look like a
#: row that was never written — so the next pass would write over it, and the
#: pass after that would retract whatever fell off the far end. There is no
#: per-pass write budget for the same reason: a partial pass that nothing
#: resumes leaves the store permanently short of the canonical file, and the
#: ledger's production shape (Emma: 358 patterns, 80 blockers) is already past
#: any cap worth naming.
READ_PAGE_SIZE = 200


class LedgerAssertionMappingError(ValueError):
    """A ledger row cannot be represented by this bounded adapter."""


def sections() -> Tuple[LedgerSection, ...]:
    """The ledger sections this adapter projects, in a stable order."""
    return _SECTIONS


def section_term(section: LedgerSection) -> str:
    try:
        return _SECTION_TERMS[section.ledger_key]
    except KeyError as error:  # pragma: no cover - guarded by _SECTIONS
        raise LedgerAssertionMappingError(
            f"no declared predicate term for ledger section {section.ledger_key!r}"
        ) from error


def ontology() -> OntologyRef:
    """Resolve the exact local ontology pin and its declared row terms.

    Resolution goes through the registry so the digest travels with the
    version: an assertion records which immutable bytes it was written
    against, and a later release cannot retroactively reinterpret it.
    """
    resource = get_knowledge_registry().resolve(ONTOLOGY_IDENTIFIER, ONTOLOGY_VERSION)
    for term in (_PATTERN_TERM, _BLOCKER_TERM):
        if f"kestrel:{term}" not in resource.selected_terms:
            raise LedgerAssertionMappingError(
                "the pinned ontology does not declare the strategy-ledger "
                f"predicate kestrel:{term}"
            )
    return OntologyRef(
        namespace=resource.namespace,
        version=str(resource.version),
        content_digest=resource.sha256,
        compatibility_profile="semantic-kb-v1",
    )


def ledger_subject(tenant_id: str) -> IRI:
    """The producing ledger, as one durable tenant-scoped instance IRI."""
    return IRI(f"urn:kestrel:agent:{tenant_id}:{_SUBJECT_SUFFIX}")


def ledger_row_object(tenant_id: str, row_id: str) -> IRI:
    """The stable row IRI an edit must not change.

    Tenant-scoped in the same ``urn:kestrel:agent:`` family the explicit-fact
    adapter already uses for instance IRIs, rather than minted under the
    ontology namespace: a vocabulary namespace names terms, and two agents'
    row ``pat_abc`` are not the same thing.
    """
    if not _ROW_ID_RE.fullmatch(row_id or ""):
        raise LedgerAssertionMappingError(
            "ledger row id is not an IRI-safe unreserved token"
        )
    return IRI(f"urn:kestrel:agent:{tenant_id}:{_OBJECT_INFIX}:{row_id}")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def row_content_digest(section: LedgerSection, row: Mapping[str, Any]) -> str:
    """Digest the row content this revision stands for.

    Includes the lifecycle fields even though a retired row is retracted
    rather than revised: the digest answers "is this row the same row it was",
    and a rule that answers it differently depending on which branch is about
    to consume it is two rules.

    Includes the row's own id for the same reason, and it is load-bearing.
    Every derived identity — revision, source occurrence, operation — descends
    from this digest, while the ASSERTION id descends from the row id. Two
    byte-identical rows are two rows (``_unique_id`` mints ``pat_X`` and
    ``pat_X-2`` precisely so one can be resolved without the other), so a
    content-only digest gave two distinct assertions one revision id and one
    operation id. That is reachable from the shipped tool: ``strategy_add_pattern``
    twice with the same text on the same day, since ``recorded_at`` is
    date-only. The second row then failed to write on every reindex, forever,
    with the store correctly refusing an operation id already spent on a
    different mutation.
    """
    fields = _SECTION_CONTENT_FIELDS[section.ledger_key]
    material = {
        "adapter_version": LEDGER_ADAPTER_VERSION,
        "section": section.ledger_key,
        "row": section.row_id(row),
        "fields": {name: str(row.get(name) or "") for name in fields},
    }
    return _canonical_digest(material)


def transition_digest(
    *, content_digest: str, predecessor_revision_id: Optional[str], action: str
) -> str:
    """Identify one transition, not one content state.

    Revision, source-occurrence and operation ids all derive from this. Keying
    them on content alone is wrong in both directions: reverting a row from B
    back to A would reuse A's immutable revision id and the write would be
    refused, and re-adding a removed row would replay the original accepted
    operation receipt and report a success that left nothing active. Naming
    the predecessor makes each distinct transition distinct while an exact
    retry of the same transition stays byte-identical, and therefore
    idempotent at the store's own receipt ledger.
    """
    return _canonical_digest(
        {
            "adapter_version": LEDGER_ADAPTER_VERSION,
            "action": action,
            "content": content_digest,
            "predecessor": predecessor_revision_id or "",
        }
    )


def revision_id(*, content_digest: str, transition: str) -> str:
    """The revision id, carrying its content digest in a parseable position.

    Reading the content digest back out of the id is what lets one paged query
    over the adapter's predicates decide "unchanged" for every row, instead of
    a provenance read per row on every reindex — and reindex runs after every
    ledger mutation.
    """
    return f"{_REVISION_PREFIX}{content_digest}:{transition}"


def content_digest_of(assertion: Assertion) -> Optional[str]:
    """The row content digest an adapter revision id encodes, if it is one."""
    value = assertion.revision_id
    if not value.startswith(_REVISION_PREFIX):
        return None
    remainder = value[len(_REVISION_PREFIX):]
    head, separator, tail = remainder.partition(":")
    if not separator or len(head) != 64 or len(tail) != 64:
        return None
    return head


def source_occurrence_id(transition: str) -> str:
    return f"{_SOURCE_PREFIX}{transition}"


def operation_id(*, action: str, transition: str) -> str:
    return f"{LEDGER_ADAPTER_VERSION}:{action}:{transition}"


@dataclass(frozen=True, slots=True)
class LedgerRowProposal:
    """The complete pinned semantic representation of one ledger row."""

    section_key: str
    row_id: str
    subject: IRI
    predicate: IRI
    object: IRI
    ontology: OntologyRef
    content_digest: str
    assertion_id: str


@dataclass(frozen=True, slots=True)
class LedgerProposalPlan:
    """What the canonical ledger says, ready for the governed executor."""

    #: The one subject every row of this tenant's ledger asserts against, so
    #: the executor's read is scoped by the same term the write uses rather
    #: than by a second derivation of it.
    subject: IRI
    proposals: Tuple[LedgerRowProposal, ...]
    #: Predicate IRI value -> section key, for scoping reconciliation.
    predicates: Mapping[str, str]
    #: Section keys whose canonical state is ambiguous; neither written nor
    #: reconciled, because a keep-set derived from an ambiguous section
    #: deletes whichever row lost the collision.
    refused_sections: Mapping[str, str]
    #: Rows that cannot be addressed as an IRI at all.
    unmappable: int
    #: Assertion ids whose canonical row is still PRESENT and active but which
    #: this pass does not assert — today, a row whose text has been blanked.
    #:
    #: They are protected from reconciliation rather than retracted. "The row
    #: is gone" and "the row is here but currently says nothing" are different
    #: facts, and conflating them is irreversible here: retraction is terminal
    #: for this adapter, so restoring the text later reports
    #: ``blocked_terminal`` forever. Absence of a claim is not a claim of
    #: absence.
    protected: Tuple[str, ...] = ()
    #: How many present rows were protected rather than asserted.
    skipped: int = 0


def map_ledger_row(
    section: LedgerSection,
    row: Mapping[str, Any],
    *,
    tenant_id: str,
    ontology_ref: OntologyRef,
) -> LedgerRowProposal:
    """Map one active, projectable ledger row to closed canonical terms."""
    from kestrel_sovereign.knowledge.assertion import derive_assertion_id

    row_id = section.row_id(row)
    subject = ledger_subject(tenant_id)
    predicate = IRI(f"{ontology_ref.namespace}{section_term(section)}")
    target = ledger_row_object(tenant_id, row_id)
    return LedgerRowProposal(
        section_key=section.ledger_key,
        row_id=row_id,
        subject=subject,
        predicate=predicate,
        object=target,
        ontology=ontology_ref,
        content_digest=row_content_digest(section, row),
        assertion_id=derive_assertion_id(
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            object=target,
        ),
    )


def build_proposal_plan(
    ledger_data: Optional[Mapping[str, Any]],
    *,
    tenant_id: str,
    ontology_ref: Optional[OntologyRef] = None,
) -> LedgerProposalPlan:
    """Read the canonical ledger into proposals, refusing ambiguous sections.

    Row ids are addresses, and ``StrategyLedger.normalize`` only disambiguates
    the ones it mints — a hand-edited file can hold two rows under one id.
    Both would map to one assertion, so whichever is written last silently
    wins and the other becomes unreachable while the projection reports
    success. The section is refused whole instead: no write, and no
    reconciliation either, because a keep-set built from an ambiguous section
    cannot distinguish "this row was removed" from "this row lost a
    collision".

    A row that is PRESENT but unassertable is handled differently from a row
    that is gone, and the distinction is load-bearing because retraction is
    terminal here:

    * **Text blanked** (``is_projectable`` false) — the row still exists and is
      still active, it just currently says nothing. Its assertion is
      ``protected`` from reconciliation rather than retracted, so restoring the
      text re-projects instead of hitting ``blocked_terminal`` forever.
    * **Row id not IRI-safe** — the section is refused whole, exactly as for
      duplicates, and for the same reason: the id is unusable, so the assertion
      it would name cannot be computed, so it cannot be protected individually
      either. Refusing is the only option that does not risk retracting it.
    * **Retired in place** (``superseded_at``/``resolved_at``) — genuinely no
      longer held, so it is deliberately absent from both sets and the
      reconciliation retracts it. That is a deliberate, tool-supported act;
      blanking a text field is not.
    """
    ontology_ref = ontology_ref or ontology()
    proposals: List[LedgerRowProposal] = []
    protected: List[str] = []
    predicates: Dict[str, str] = {}
    refused: Dict[str, str] = {}
    unmappable = 0
    skipped = 0

    data = dict(ledger_data or {})
    for section in _SECTIONS:
        predicate = IRI(f"{ontology_ref.namespace}{section_term(section)}")
        predicates[predicate.value] = section.ledger_key
        # Every row, not only the projectable ones: a blank-texted row still
        # occupies its id, so a duplicate between it and an asserted row is the
        # same ambiguity a duplicate between two asserted rows is.
        rows = section.rows(data)

        # A section member that is not a mapping is silently dropped by
        # ``_dict_rows``, so it is invisible to BOTH classifications while its
        # section stays in scope for reconciliation. If that member used to be
        # a well-formed row, its assertion is then terminally retracted by a
        # hand edit that mangled one line. Refuse the section instead, exactly
        # as for an unusable id.
        raw = data.get(section.ledger_key)
        if isinstance(raw, list) and len(raw) != len(rows):
            unmappable += len(raw) - len(rows)
            refused[section.ledger_key] = "malformed_rows"
            continue

        # An id-less row is addressed by a digest of its own TEXT, so editing
        # that text moves its address and orphans the assertion written under
        # the old one — terminally. ``StrategyLedger.normalize`` mints ids and
        # the feature persists them before projecting, so this is unreachable
        # on the healthy path; it becomes reachable when that save failed. The
        # keep-set must not depend on a write having succeeded elsewhere.
        if any(not str(row.get("id") or "").strip() for row in rows):
            refused[section.ledger_key] = "unaddressed_rows"
            continue

        seen: Dict[str, int] = {}
        for row in rows:
            seen[section.row_id(row)] = seen.get(section.row_id(row), 0) + 1
        duplicates = sorted(key for key, count in seen.items() if count > 1)
        if duplicates:
            refused[section.ledger_key] = "duplicate_row_ids"
            continue
        unusable = [
            row for row in rows
            if not _ROW_ID_RE.fullmatch(section.row_id(row) or "")
        ]
        if unusable:
            unmappable += len(unusable)
            refused[section.ledger_key] = "unmappable_row_ids"
            continue
        for row in rows:
            if not section.is_active(row):
                # A retired row is still a row; it simply no longer supports
                # the claim. Reconciliation retracts its assertion by way of
                # the keep-set, so it is deliberately absent from here.
                continue
            mapped = map_ledger_row(
                section, row, tenant_id=tenant_id, ontology_ref=ontology_ref
            )
            if not section.is_projectable(row):
                # Present, active, and currently says nothing. Keep it; do not
                # assert it, and above all do not retract it.
                protected.append(mapped.assertion_id)
                skipped += 1
                continue
            proposals.append(mapped)
    return LedgerProposalPlan(
        subject=ledger_subject(tenant_id),
        proposals=tuple(proposals),
        predicates=predicates,
        refused_sections=refused,
        unmappable=unmappable,
        protected=tuple(protected),
        skipped=skipped,
    )


def build_source(
    proposal: LedgerRowProposal,
    *,
    transition: str,
    owning_agent_id: str,
) -> SourceOccurrence:
    """Build provenance from the trusted turn context, never from row text.

    The locator names the canonical file and the row's address, which is a
    truthful pointer that carries no pattern or blocker prose.
    """
    provenance = current_invocation_provenance()
    stem = f"{_LOCATOR_STEM}{proposal.section_key}:{proposal.row_id}"
    if provenance is None:
        source_kind = _SOURCE_KIND
        locator = stem
        actor = owning_agent_id
        observed_at = datetime.now(timezone.utc).isoformat()
    else:
        source_kind = provenance.source_kind
        locator = f"{provenance.source_locator}#{stem}"
        actor = provenance.actor or owning_agent_id
        observed_at = provenance.received_at
    return SourceOccurrence(
        source_occurrence_id=source_occurrence_id(transition),
        source_kind=source_kind,
        locator=locator,
        received_at=observed_at,
        content_digest=f"sha256:{proposal.content_digest}",
        actor=actor,
        selector=_SELECTOR,
    )


def build_assertion(
    *,
    binding: SemanticAssertionBinding,
    proposal: LedgerRowProposal,
    source: SourceOccurrence,
    transition: str,
    supersedes_revision_id: Optional[str] = None,
) -> Assertion:
    """Build one proposal after the privacy wrapper has bound its metadata."""
    return Assertion(
        tenant_id=binding.tenant_id,
        owning_agent_id=binding.owning_agent_id,
        subject=proposal.subject,
        predicate=proposal.predicate,
        object=proposal.object,
        revision_id=revision_id(
            content_digest=proposal.content_digest, transition=transition
        ),
        confidence=Decimal("1"),
        confidence_method=_CONFIDENCE_METHOD,
        confidence_basis=_CONFIDENCE_BASIS,
        # The agent recorded the row itself, in its own canonical ledger. That
        # is an assertion by this agent, not a report of someone else's.
        epistemic_state=EpistemicState.ASSERTED,
        asserted_at=source.received_at,
        ontology_version=proposal.ontology,
        lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification=binding.privacy_classification,
        release_policy_reference=binding.release_policy_reference,
        visibility=binding.visibility,
        supersedes_revision_id=supersedes_revision_id,
    )


def declares_adapter_markers(
    assertion: Assertion, ontology_ref: OntologyRef
) -> bool:
    """Whether an assertion *claims* to be one of ours.

    Necessary, never sufficient. Any canonical writer can supply these
    strings, so nothing is superseded or retracted on their evidence alone —
    :func:`is_adapter_source` checks the provenance grammar before any
    mutation.

    Deliberately says nothing about the revision id. Ownership ("did this
    adapter author this assertion") and content ("which row state does this
    revision stand for") are two questions, and a retraction answers them
    differently: the store mints its own opaque revision id for the tombstone
    while preserving the markers and provenance. Folding the revision shape in
    here made the adapter's OWN retracted assertion unrecognizable, so
    re-adding a hand-deleted row reported ``foreign`` — a claim that some other
    writer owns the subject, which was false and would have sent an operator
    looking for a competing producer that does not exist. Content is answered
    separately by :func:`content_digest_of`, which returns ``None`` rather than
    a wrong digest for any revision it did not write.
    """
    return (
        assertion.confidence_method == _CONFIDENCE_METHOD
        and assertion.confidence_basis == _CONFIDENCE_BASIS
        and assertion.ontology_version == ontology_ref
        and isinstance(assertion.lineage, DirectLineage)
    )


def is_adapter_source(source: SourceOccurrence) -> bool:
    """Recognize the bounded provenance grammar owned by this adapter."""
    is_startup_locator = (
        source.source_kind == _SOURCE_KIND
        and source.locator.startswith(_LOCATOR_STEM)
    )
    is_request_locator = (
        source.source_kind != _SOURCE_KIND
        and f"#{_LOCATOR_STEM}" in source.locator
    )
    return (
        source.source_occurrence_id.startswith(_SOURCE_PREFIX)
        and (is_startup_locator or is_request_locator)
        and source.selector == _SELECTOR
    )


def has_adapter_provenance(sources: Iterable[SourceOccurrence]) -> bool:
    return any(is_adapter_source(source) for source in sources)


@dataclass
class LedgerAssertionReport:
    """A content-free outcome: counts and reasons, never row text.

    Callers surface this to logs and to ``!strategy`` output, so it must stay
    free of pattern and blocker prose.
    """

    projected: int = 0
    revised: int = 0
    unchanged: int = 0
    retracted: int = 0
    #: Present rows this pass deliberately did not assert and did not retract.
    skipped: int = 0
    foreign: int = 0
    blocked_terminal: int = 0
    unmappable: int = 0
    failed: int = 0
    refused_sections: Dict[str, str] = field(default_factory=dict)
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "projected": self.projected,
            "revised": self.revised,
            "unchanged": self.unchanged,
            "retracted": self.retracted,
            "skipped": self.skipped,
            "foreign": self.foreign,
            "blocked_terminal": self.blocked_terminal,
            "unmappable": self.unmappable,
            "failed": self.failed,
        }
        if self.refused_sections:
            data["refused_sections"] = dict(self.refused_sections)
        if self.skipped_reason:
            data["skipped_reason"] = self.skipped_reason
        return data

    @property
    def needs_attention(self) -> bool:
        return bool(
            self.failed
            or self.foreign
            or self.blocked_terminal
            or self.unmappable
            or self.refused_sections
            or self.skipped_reason
        )


def statuses_for_current_read() -> Tuple[str, ...]:
    """Every status a *current* adapter revision can carry.

    ``AssertionQuery`` defaults to active-only. Reading the terminal states
    too is what makes ``blocked_terminal`` observable instead of being
    misread as "no assertion yet" and turned into a replayed write that
    reports success while nothing becomes active.
    """
    from kestrel_sovereign.knowledge import AssertionStatus

    return (
        AssertionStatus.ACTIVE,
        AssertionStatus.RETRACTED,
        AssertionStatus.QUARANTINED,
        AssertionStatus.DELETED,
        AssertionStatus.SUPERSEDED,
    )


def summarize(reports: Sequence[LedgerAssertionReport]) -> LedgerAssertionReport:  # pragma: no cover - convenience
    total = LedgerAssertionReport()
    for item in reports:
        total.projected += item.projected
        total.revised += item.revised
        total.unchanged += item.unchanged
        total.retracted += item.retracted
        total.skipped += item.skipped
        total.foreign += item.foreign
        total.blocked_terminal += item.blocked_terminal
        total.unmappable += item.unmappable
        total.failed += item.failed
        total.refused_sections.update(item.refused_sections)
    return total
