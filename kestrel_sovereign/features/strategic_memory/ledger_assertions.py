"""Project strategy-ledger rows into canonical semantic assertions (#3051).

``semantic_assertions`` was built, versioned and governed, and held zero rows:
nothing wrote to it, so there was no semantic recall path for structured agent
knowledge at all. This is the producer that fills it, and it is deliberately
the narrowest one that answers the finding: the strategic-memory ledger's
patterns and blockers.

**The unit is the claim.** Not a chunk, not a subgraph. A pattern row already
says one thing and a blocker row already says one thing, so each renders to
exactly one assertion whose object is the rendered claim text. That is the same
granularity ``memory_episodes`` uses -- a synthesized claim, not raw structure
-- and it is why this does not become an embedding column on ``graph_nodes``:
the graph stays the structural index and semantics stay here.

**The direction of truth is #2851's, unchanged.** ``STRATEGY_LEDGER.yaml`` is
canonical; the assertion is derived and never writes back. Three rules follow:

1. **Identity is the row, not the text.** The subject IRI is a function of the
   row's durable ``id``, so an edited row keeps its subject. Since assertion
   identity hashes ``tenant + subject + predicate + object``, an edit changes
   the object and therefore the assertion id -- which is exactly why the edit
   is committed as a *supersession* of the predecessor found at that stable
   subject, rather than as a second competing claim. Re-projection upserts; an
   edit re-projects; neither duplicates.
2. **Re-rendering is keyed on a revision digest.** :func:`claim_revision_digest`
   covers the adapter version, the renderer version, the row id and the
   rendered claim. An unchanged row produces the same digest and the pass does
   nothing; a changed row -- or a changed renderer -- produces a new one and
   re-projects.
3. **Removal reconciles.** A row deleted from the canonical file has its
   assertion retracted, never left behind. A derived index that keeps claims
   the file no longer holds is the defect #2851 and #2954 both existed to fix.

**Lineage and owner are set at write time, not backfilled** (#3060). The writer
knows its own provenance -- which file, which row, which digest -- and declares
it in ``DirectLineage`` plus a ``SourceOccurrence``. ``owning_agent_id`` comes
from the wrapper-issued binding, never from a caller. This is what makes #3060
answerable: the same content now carries both ``claim_source`` on the graph
node and a real lineage record here, so the two vocabularies can finally be
compared against each other. Nothing here deprecates or narrows ``claim_source``
-- that decision belongs to #3060, after the comparison, not before it.

**Out of scope, deliberately:** the vector projection. Populating
``semantic_assertion_vector_projection_entries`` needs an embedding-provider
binding and a pinned ``SemanticVectorProfile`` (capability digest, dimension,
renderer version) -- a separate governed artifact with its own release-evidence
obligations. Canonical writes come first; a vector is disposable acceleration
data over them.

Best-effort throughout, for the same reason :mod:`ledger_index` is: the
canonical record is already on disk, so a failure to derive an index must never
turn into a strategic-memory failure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as replace_assertion
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote
import hashlib
import json
import logging
import unicodedata

from .ledger_index import BLOCKER_SECTION, PATTERN_SECTION, LedgerSection

logger = logging.getLogger(__name__)

#: The closed grammar this producer writes under. It is recorded as every
#: assertion's ``confidence_method``, which is what makes "did this adapter
#: write this row?" answerable without a side table.
STRATEGY_ASSERTION_ADAPTER_VERSION = "strategic-memory-ledger-assertion-v1"

#: Pinned separately from the adapter version because a rendering change is a
#: re-projection trigger on its own: the claim text IS the assertion object, so
#: changing how a row renders changes what was asserted.
STRATEGY_CLAIM_RENDERER_VERSION = "strategy-ledger-claim-v1"

ONTOLOGY_IDENTIFIER = "kestrel-vocab"
ONTOLOGY_VERSION = "1.2.0"

#: Declares this projection's fidelity to the canonical row -- NOT epistemic
#: certainty about the claim's content, which the ledger does not record. The
#: pair of confidence fields is where that distinction is stated, so confidence
#: 1 here means "this is exactly what the ledger holds".
_CONFIDENCE = Decimal("1")
_CONFIDENCE_BASIS = "canonical-ledger-row"

_SOURCE_KIND = "strategy_ledger_row"

#: Ceiling on the current-assertion read below. Reconciliation retracts rows,
#: so it must never run against a partial view of what exists -- a saturated
#: read is reported as a check that did not run, exactly as
#: ``ledger_index.MEMBERSHIP_READ_CAP`` does for the graph.
_READ_PAGE_SIZE = 500
_MAX_READ_PAGES = 40

#: Governed writes are expensive (each carries a full SHACL validation of the
#: tentative post-state), and the first projection of a real ledger is a few
#: hundred of them. Bounding one pass keeps a single tool call from stalling;
#: progress is monotonic because unchanged rows are skipped, so the next pass
#: continues where this one stopped rather than redoing it.
_MAX_WRITES_PER_PASS = 200


class LedgerAssertionError(RuntimeError):
    """This producer cannot map or govern a ledger row's claim."""


@dataclass(frozen=True)
class AssertionSection:
    """One ledger row kind, bound to the claim it asserts.

    Wraps the existing :class:`~.ledger_index.LedgerSection` rather than
    restating it. Which rows are projectable, how a row id is derived and
    whether a row is active are already answered there, and #3064 is the record
    of what happens when two call sites answer one question separately.
    """

    ledger: LedgerSection
    kind: str
    predicate_term: str
    render: Callable[[Dict[str, Any]], str]

    @property
    def noun(self) -> str:
        return self.ledger.noun


def _clean(value: object) -> str:
    """Collapse a ledger field to canonical single-spaced NFC text.

    The claim text is the assertion's object, so it is part of the identity
    hash. Re-indenting a YAML block scalar must not mint a new assertion.
    """
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.split())


def render_pattern_claim(row: Dict[str, Any]) -> str:
    """Render one pattern row as a single claim.

    The implication is part of the claim, not context: "X, which implies Y" is
    what a pattern row asserts, and an edit to the implication is an edit to
    the claim. It therefore re-projects as a revision -- which is only possible
    because identity is keyed on the row, not on this text.
    """
    claim = _clean(row.get("pattern"))
    implication = _clean(row.get("implication"))
    if implication:
        return f"{claim} Implication: {implication}"
    return claim


def render_blocker_claim(row: Dict[str, Any]) -> str:
    """Render one blocker row as a single claim.

    ``#42`` names an issue only relative to a repository, so the repo travels
    with it or not at all -- the same reason :mod:`ledger_index` keeps ``repo``
    on the node.
    """
    parts: List[str] = [_clean(row.get("title"))]
    issue = _clean(row.get("issue"))
    repo = _clean(row.get("repo"))
    if issue:
        reference = f"{repo} {issue}" if repo else issue
        parts.append(f"Blocked on {reference}.")
    severity = _clean(row.get("severity"))
    if severity:
        parts.append(f"Severity: {severity}.")
    owner = _clean(row.get("owner"))
    if owner:
        parts.append(f"Owner: {owner}.")
    return " ".join(part for part in parts if part)


PATTERN_ASSERTIONS = AssertionSection(
    ledger=PATTERN_SECTION,
    kind="pattern",
    predicate_term="strategicPattern",
    render=render_pattern_claim,
)

BLOCKER_ASSERTIONS = AssertionSection(
    ledger=BLOCKER_SECTION,
    kind="blocker",
    predicate_term="strategicBlocker",
    render=render_blocker_claim,
)

SECTIONS: Tuple[AssertionSection, ...] = (PATTERN_ASSERTIONS, BLOCKER_ASSERTIONS)


def _canonical_digest(material: Dict[str, Any]) -> str:
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def claim_revision_digest(
    *, tenant_id: str, kind: str, row_id: str, claim: str
) -> str:
    """The digest a re-projection is keyed on.

    Covers the adapter and renderer versions as well as the content, so a
    change to either re-projects rather than silently leaving stale text
    asserted under a new rendering contract.
    """
    return _canonical_digest(
        {
            "adapter_version": STRATEGY_ASSERTION_ADAPTER_VERSION,
            "claim": claim,
            "kind": kind,
            "renderer_version": STRATEGY_CLAIM_RENDERER_VERSION,
            "row_id": row_id,
            "tenant_id": tenant_id,
        }
    )


def claim_subject_iri(tenant_id: str, kind: str, row_id: str) -> str:
    """The stable subject a row's claim is asserted about.

    Percent-encoded because a ledger id is hand-editable: encoding is injective
    and always valid, where rejecting an odd id would make that row silently
    unreachable. ``quote(safe="")`` never escapes an unreserved character, so
    RFC 3986 normalization leaves the result alone and identity stays stable.
    """
    return (
        f"urn:kestrel:agent:{tenant_id}:strategy:"
        f"{kind}:{quote(row_id, safe='')}"
    )


def _ontology():
    """Resolve the exact ontology pin and prove it declares our predicates."""
    from kestrel_sovereign.knowledge import OntologyRef
    from kestrel_sovereign.knowledge.registry import get_knowledge_registry

    resource = get_knowledge_registry().resolve(
        ONTOLOGY_IDENTIFIER, ONTOLOGY_VERSION
    )
    required = {
        f"kestrel:{section.predicate_term}" for section in SECTIONS
    }
    missing = sorted(required - set(resource.selected_terms))
    if missing:
        raise LedgerAssertionError(
            "the pinned ontology does not declare the strategic-memory "
            f"predicates: {', '.join(missing)}"
        )
    return OntologyRef(
        namespace=resource.namespace,
        version=str(resource.version),
        content_digest=resource.sha256,
        compatibility_profile="semantic-kb-v1",
    )


@dataclass(frozen=True)
class ClaimProposal:
    """Everything one row's claim needs, computed before any store call."""

    section: AssertionSection
    row_id: str
    subject_value: str
    predicate_value: str
    claim: str
    digest: str

    @property
    def revision_id(self) -> str:
        return f"revision:{STRATEGY_ASSERTION_ADAPTER_VERSION}:{self.digest}"

    @property
    def source_occurrence_id(self) -> str:
        return f"source:{STRATEGY_ASSERTION_ADAPTER_VERSION}:{self.digest}"

    def operation_id(self, action: str) -> str:
        return f"{STRATEGY_ASSERTION_ADAPTER_VERSION}:{action}:{self.digest}"


def build_proposals(
    section: AssertionSection,
    rows: List[Dict[str, Any]],
    *,
    tenant_id: str,
    ontology_namespace: str,
) -> List[ClaimProposal]:
    """Map projectable rows to claim proposals, skipping text-less rows.

    ``is_projectable`` is the ledger index's rule, reused: a row with no text
    has nothing to reason over, and answering that question a second way is how
    the reader and the writer come to disagree about what the index holds.
    """
    predicate_value = f"{ontology_namespace}{section.predicate_term}"
    proposals: List[ClaimProposal] = []
    for row in rows:
        if not section.ledger.is_projectable(row):
            continue
        claim = section.render(row)
        if not claim:
            continue
        row_id = section.ledger.row_id(row)
        proposals.append(
            ClaimProposal(
                section=section,
                row_id=row_id,
                subject_value=claim_subject_iri(tenant_id, section.kind, row_id),
                predicate_value=predicate_value,
                claim=claim,
                digest=claim_revision_digest(
                    tenant_id=tenant_id,
                    kind=section.kind,
                    row_id=row_id,
                    claim=claim,
                ),
            )
        )
    return proposals


def _adapter_owned(assertion, ontology) -> bool:
    """Only this adapter's own claims are ours to supersede or retract.

    A claim written by anything else at the same subject is a foreign
    canonical record. The projection reports it and leaves it alone rather
    than overwriting a record it did not create.
    """
    return (
        assertion.confidence_method == STRATEGY_ASSERTION_ADAPTER_VERSION
        and assertion.confidence_basis == _CONFIDENCE_BASIS
        and assertion.ontology_version.namespace == ontology.namespace
    )


async def _read_current(storage, predicate_value: str, report: Dict[str, Any]):
    """Page every current assertion at one predicate for this tenant.

    Returns ``(by_subject, complete)``. ``complete`` is False when the read
    saturated its page budget, and the caller must then skip reconciliation:
    retracting against a partial view deletes claims the ledger still holds.
    """
    from kestrel_sovereign.knowledge import AssertionQuery, IRI

    predicate = IRI(predicate_value)
    by_subject: Dict[str, Any] = {}
    cursor: Optional[str] = None
    for _ in range(_MAX_READ_PAGES):
        page = await storage.query_assertions(
            AssertionQuery(
                predicate=predicate,
                limit=_READ_PAGE_SIZE,
                cursor=cursor,
            )
        )
        for assertion in page:
            by_subject[assertion.subject.value] = assertion
        if len(page) < _READ_PAGE_SIZE:
            return by_subject, True
        cursor = page[-1].revision_id
    report["read_saturated"] = True
    return by_subject, False


def _build_assertion(proposal: ClaimProposal, binding, ontology, source):
    from kestrel_sovereign.knowledge import (
        Assertion,
        DirectLineage,
        EpistemicState,
        IRI,
        Literal,
        XSD_STRING,
    )

    return Assertion(
        tenant_id=binding.tenant_id,
        owning_agent_id=binding.owning_agent_id,
        subject=IRI(proposal.subject_value),
        predicate=IRI(proposal.predicate_value),
        object=Literal(proposal.claim, XSD_STRING),
        revision_id=proposal.revision_id,
        confidence=_CONFIDENCE,
        confidence_method=STRATEGY_ASSERTION_ADAPTER_VERSION,
        confidence_basis=_CONFIDENCE_BASIS,
        epistemic_state=EpistemicState.REPORTED,
        asserted_at=source.received_at,
        ontology_version=ontology,
        lineage=DirectLineage((source.source_occurrence_id,)),
        privacy_classification=binding.privacy_classification,
        release_policy_reference=binding.release_policy_reference,
        visibility=binding.visibility,
    )


def _build_source(proposal: ClaimProposal, binding, ledger_locator: str):
    from kestrel_sovereign.knowledge import SourceOccurrence

    return SourceOccurrence(
        source_occurrence_id=proposal.source_occurrence_id,
        source_kind=_SOURCE_KIND,
        # Names the canonical record this claim was derived from -- the file
        # section and the row, not a conversation message. A locator that
        # cannot be followed back to the source is not provenance.
        locator=f"{ledger_locator}#{proposal.section.ledger.ledger_key}:{proposal.row_id}",
        received_at=datetime.now(timezone.utc).isoformat(),
        content_digest=f"sha256:{proposal.digest}",
        actor=binding.owning_agent_id,
        selector=proposal.section.kind,
    )


async def _project_section(
    storage,
    section: AssertionSection,
    rows: List[Dict[str, Any]],
    *,
    binding,
    ontology,
    ledger_locator: str,
    report: Dict[str, Any],
    write_budget: List[int],
) -> None:
    proposals = build_proposals(
        section,
        rows,
        tenant_id=binding.tenant_id,
        ontology_namespace=ontology.namespace,
    )
    report["skipped"] += sum(
        1 for row in rows if not section.ledger.is_projectable(row)
    )

    current, complete = await _read_current(
        storage, f"{ontology.namespace}{section.predicate_term}", report
    )

    for proposal in proposals:
        if write_budget[0] <= 0:
            report["deferred"] += 1
            report["budget_exhausted"] = True
            continue
        existing = current.get(proposal.subject_value)
        if existing is not None and not _adapter_owned(existing, ontology):
            # Someone else's canonical claim sits at this subject. Refusing is
            # the only safe move: superseding it would make this projection an
            # editor of records it does not own.
            report["foreign"] += 1
            continue
        if existing is not None and existing.object.value == proposal.claim:
            report["unchanged"] += 1
            continue

        source = _build_source(proposal, binding, ledger_locator)
        assertion = _build_assertion(proposal, binding, ontology, source)
        try:
            if existing is None:
                result = await storage.put_assertion(
                    assertion,
                    source_occurrences=(source,),
                    operation_id=proposal.operation_id("put"),
                )
                outcome = "projected"
            else:
                result = await storage.supersede_assertion(
                    existing.revision_id,
                    replace_assertion(
                        assertion,
                        supersedes_revision_id=existing.revision_id,
                    ),
                    source_occurrences=(source,),
                    operation_id=proposal.operation_id("supersede"),
                )
                outcome = "revised"
        except Exception as error:  # noqa: BLE001 - an index never fails a write
            logger.debug(
                "ledger assertion projection failed for %s row %s: %s",
                section.kind,
                proposal.row_id,
                error,
            )
            report["failed"] += 1
            continue
        write_budget[0] -= 1
        if not getattr(result, "accepted", False):
            # A rejected governed write is not a silent no-op: the claim is
            # absent and the next pass must try again, so it is counted as a
            # failure rather than as a projection.
            report["rejected"] += 1
            continue
        report[outcome] += 1

    if not complete:
        # Reconciliation without a complete view retracts claims the ledger
        # still holds. Report the unrun check instead.
        report["reconcile_skipped"] = "read_saturated"
        return
    await _retract_orphans(
        storage,
        section,
        current,
        {proposal.subject_value for proposal in proposals},
        ontology=ontology,
        report=report,
        write_budget=write_budget,
    )


async def _retract_orphans(
    storage,
    section: AssertionSection,
    current: Dict[str, Any],
    keep: set,
    *,
    ontology,
    report: Dict[str, Any],
    write_budget: List[int],
) -> None:
    """Retract this adapter's claims for rows the canonical file dropped.

    Retracted, not deleted: the ledger stopped asserting the claim, which is a
    withdrawal with a history, not an erasure.
    """
    for subject_value, assertion in current.items():
        if subject_value in keep:
            continue
        if not _adapter_owned(assertion, ontology):
            continue
        if write_budget[0] <= 0:
            report["deferred"] += 1
            report["budget_exhausted"] = True
            continue
        try:
            await storage.retract_assertion(
                assertion.assertion_id,
                assertion.revision_id,
                operation_id=(
                    f"{STRATEGY_ASSERTION_ADAPTER_VERSION}:retract:"
                    f"{assertion.revision_id}"
                ),
            )
        except Exception as error:  # noqa: BLE001
            logger.debug(
                "ledger assertion reconcile could not retract %s claim: %s",
                section.kind,
                error,
            )
            report["failed"] += 1
            continue
        write_budget[0] -= 1
        report["retracted"] += 1


def _empty_report() -> Dict[str, Any]:
    return {
        "projected": 0,
        "revised": 0,
        "unchanged": 0,
        "retracted": 0,
        "skipped": 0,
        "failed": 0,
        "rejected": 0,
        "foreign": 0,
        "deferred": 0,
    }


async def project_ledger_assertions(
    storage: Any,
    ledger: Any,
    *,
    ledger_locator: str = "strategy-ledger",
) -> Dict[str, Any]:
    """Write the ledger's patterns and blockers as canonical assertions.

    Takes the LEDGER, not its ``data``, for the reason
    :func:`~.ledger_index.project_ledger` does: reconciliation derives its
    keep-set from the rows it is given, so a failed parse -- which leaves the
    sections empty -- would read as "every row was deleted" and retract the
    whole derived set. A bare mapping cannot express the difference between
    "no rows" and "could not be read", so readability has to travel with the
    rows.

    Returns a content-free report: counts and reasons only, never claim text,
    since callers surface it to logs.
    """
    report = _empty_report()
    if isinstance(ledger, Mapping):
        ledger_data: Dict[str, Any] = dict(ledger)
    else:
        if not getattr(ledger, "readable", True):
            report["skipped_reason"] = "ledger_unavailable"
            return report
        ledger_data = getattr(ledger, "data", {}) or {}

    total = sum(
        len(section.ledger.rows(ledger_data)) for section in SECTIONS
    )
    if storage is None:
        report["skipped"] = total
        report["skipped_reason"] = "no_storage"
        return report

    binding_factory = getattr(
        storage, "governed_semantic_assertion_binding", None
    )
    if binding_factory is None:
        # Raw AsyncStorage reports `normal`/PRIVATE unconditionally because it
        # cannot see the live privacy policy. A producer must not take those
        # fields from it, so an unwrapped storage is a skip, not a downgrade.
        report["skipped"] = total
        report["skipped_reason"] = "no_governed_binding"
        return report

    try:
        binding = binding_factory("strategy_ledger_assertion")
    except Exception as error:  # noqa: BLE001 - privacy refusal is not a bug
        logger.debug("ledger assertion projection not permitted: %s", error)
        report["skipped"] = total
        report["skipped_reason"] = "privacy_denied"
        return report

    try:
        ontology = _ontology()
    except Exception as error:  # noqa: BLE001
        logger.warning("ledger assertion projection has no ontology pin: %s", error)
        report["skipped"] = total
        report["skipped_reason"] = "ontology_unavailable"
        return report

    write_budget = [_MAX_WRITES_PER_PASS]
    for section in SECTIONS:
        try:
            await _project_section(
                storage,
                section,
                section.ledger.rows(ledger_data),
                binding=binding,
                ontology=ontology,
                ledger_locator=ledger_locator,
                report=report,
                write_budget=write_budget,
            )
        except Exception as error:  # noqa: BLE001 - never fail a YAML write
            logger.warning(
                "ledger assertion projection failed for %s: %s",
                section.noun,
                error,
            )
            report["failed"] += 1
            report.setdefault("section_errors", []).append(section.noun)
    return report
