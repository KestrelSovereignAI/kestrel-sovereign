"""Apply the established edition-one amendment procedure to fixed guarantees."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .amendment_artifact import AmendmentArtifactError
from .emancipation import (
    EmancipationContract, extract_amendment_viii, matches_recorded_emancipation_contract, render_amendment_viii,
    v1_platform_text,
)
from .structure import constitution_version, legacy_section_content, markdown_sections_at_level
from .v1_releases import AmendmentRule, CURRENT_V1_RELEASE, V1Release, V1_RELEASES, published_v1_release

LEGACY_FIXED_AMENDMENT_FIELD = "legacy_fixed_amendment"
LEGACY_FIXED_AMENDMENT_PURPOSE = "kestrel.constitution.amend-fixed-guarantees.v1"
_BOOK_I = "Book I: Universal Values"
_BOOK_II = "Book II: The Sovereign Amendments"
_BOOK_FIXED_HEADINGS = frozenset({"Preamble", _BOOK_I, _BOOK_II})
_FOUNDING_ARTICLE = re.compile(r"^Article (?:I|II|III|IV|V): ")
_ARTICLE_V = "Article V: The Amendment Process"
_AMENDMENT_HEADINGS = frozenset({"The Amendment Process", _ARTICLE_V})
_FIXED_HEADINGS = _BOOK_FIXED_HEADINGS | _AMENDMENT_HEADINGS
# The f0672b394 edition expressly delegates its Preamble and procedure to
# Genesis. The following publication instead makes those platform provisions.
_PUBLISHER_PROVISIONS = {
    AmendmentRule.FOUNDING_ARTICLES: frozenset(),
    AmendmentRule.RANKED_BOOKS: _FIXED_HEADINGS - {_BOOK_II},
    AmendmentRule.SOVEREIGN_FRAME: frozenset({_BOOK_I}),
    AmendmentRule.PLATFORM_AUTHORED_BOOKS: _FIXED_HEADINGS,
}


def _fixed_provisions(text: str) -> dict[str, tuple[str, ...]]:
    provisions: dict[str, tuple[str, ...]] = {}
    for heading, span in markdown_sections_at_level(text, 2):
        if heading in _FIXED_HEADINGS:
            provisions[heading] = (*provisions.get(heading, ()), legacy_section_content(text, span))
    return provisions


def _matches_published_provisions(
    proposed_text: str, contract: EmancipationContract | None, publication: V1Release, provisions: frozenset[str],
) -> bool:
    published = publication.authenticated_text()
    if contract is not None and contract.enabled:
        if not publication.supports_activation or not matches_recorded_emancipation_contract(proposed_text, contract):
            return False
        slot = extract_amendment_viii(published)
        if slot is None:
            raise AmendmentArtifactError("Edition 1 publication lacks its registered emancipation provision")
        proposed_text = v1_platform_text(proposed_text, contract, replacement=slot)
    expected, proposed = _fixed_provisions(published), _fixed_provisions(proposed_text)
    return all(expected.get(heading) == proposed.get(heading) for heading in provisions)


# A caption introduces the enumerated rules. Only normalize a plain lead-in
# immediately before rule 1. Conditions, additional paragraphs and the rules
# themselves still participate in the authority comparison.
_RULE_LIST_CAPTION = re.compile(
    r"^(?:this|the) constitution (?:can|may) be amended "
    r"(?:according to|in accordance with|under) the following rules:[ \t]*\r?\n"
    r"(?=\r?\n1\. \*\*Book I\*\*)", re.IGNORECASE | re.MULTILINE,
)


def _procedure_authority(provisions: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    return {
        heading: tuple(_RULE_LIST_CAPTION.sub("", text, count=1) for text in provisions[heading])
        for heading in _AMENDMENT_HEADINGS & provisions.keys()
    }


def _amendment_rule(
    prior_hash: str | None, provisions: dict[str, tuple[str, ...]], headings: set[str],
) -> AmendmentRule | None:
    release = published_v1_release(prior_hash) if prior_hash is not None else None
    if release is not None:
        return release.amendment_rule
    if _ARTICLE_V in headings and sum(bool(_FOUNDING_ARTICLE.match(heading)) for heading in headings) > 1:
        return AmendmentRule.FOUNDING_ARTICLES
    if not _BOOK_FIXED_HEADINGS <= headings or not _AMENDMENT_HEADINGS & headings:
        return None
    # A derived ranked text can inherit the older rule only if its operative
    # amendment procedure still matches a published older procedure exactly.
    procedure = _procedure_authority(provisions)
    for release in V1_RELEASES:
        if release.amendment_rule in (AmendmentRule.RANKED_BOOKS, AmendmentRule.SOVEREIGN_FRAME):
            published = _fixed_provisions(release.authenticated_text())
            if procedure == _procedure_authority(published):
                return release.amendment_rule
    return AmendmentRule.PLATFORM_AUTHORED_BOOKS


def verify_legacy_fixed_guarantee_change(
    artifact: Mapping, *, sovereign_document: Mapping, target_did: str,
    prior_hash: str | None, prior_text: str | None, new_hash: str, new_text: str,
    identity_directory: Path | None,
    anchored_contract: EmancipationContract | None = None,
    proposed_contract: EmancipationContract | None = None,
    hosted_identity_evidence_path: Path | None = None,
    active_signing_did: str | None = None,
) -> dict:
    """Preserve legacy repair while requiring Genesis for material edition-one amendments."""
    if prior_hash == new_hash or constitution_version(new_text) != 1:
        return {}
    release = published_v1_release(prior_hash) if prior_hash is not None else None
    if prior_text is None:
        if release is None:
            return {}
        prior_text = release.authenticated_text()
    if hashlib.sha256(prior_text.encode()).hexdigest() != prior_hash:
        raise AmendmentArtifactError("Prior governing bytes do not match their authoritative digest")
    if constitution_version(prior_text) == 2:
        raise AmendmentArtifactError("Cannot replace an adopted edition 2 Constitution with edition 1")
    prior_platform = v1_platform_text(prior_text, anchored_contract, replacement=render_amendment_viii(None))
    before = _fixed_provisions(prior_platform)
    headings = {heading for heading, _ in markdown_sections_at_level(prior_platform, 2)}
    rule = _amendment_rule(prior_hash, before, headings)
    if rule is None:
        return {}
    founding = rule is AmendmentRule.FOUNDING_ARTICLES
    contract = proposed_contract or anchored_contract
    proposed_platform = new_text
    if contract is not None and contract.enabled:
        # The delegation belongs to the governing platform text. A personal
        # role edit changes its digest but preserves the published dormant slot.
        slot = extract_amendment_viii(prior_platform)
        if slot is None or not any(
            slot == extract_amendment_viii(item.authenticated_text())
            for item in V1_RELEASES if item.supports_activation
        ):
            slot = render_amendment_viii(None)
        proposed_platform = v1_platform_text(new_text, contract, replacement=slot)
    after = _fixed_provisions(proposed_platform)
    if not founding and before == after:
        return {}
    changed = {heading for heading in _FIXED_HEADINGS if before.get(heading) != after.get(heading)}
    genesis = {}
    # Only the two published procedures that assign Book I to the framework
    # explicitly allow a publisher update by ordinary Sovereign reanchor.
    if rule in (AmendmentRule.FOUNDING_ARTICLES, AmendmentRule.RANKED_BOOKS) or changed - {_BOOK_I}:
        genesis = _verify_legacy_declaration(
            artifact, sovereign_document, target_did, prior_hash, new_hash, identity_directory,
            hosted_identity_evidence_path, active_signing_did,
            "this founding Constitution" if founding else "edition 1 fixed guarantees",
        )
    platform = _PUBLISHER_PROVISIONS[rule]
    if changed & platform:
        sources = (
            (CURRENT_V1_RELEASE,) if rule is AmendmentRule.PLATFORM_AUTHORED_BOOKS
            else (item for item in V1_RELEASES if item.amendment_rule is not AmendmentRule.FOUNDING_ARTICLES)
        )
        if not any(_matches_published_provisions(new_text, contract, item, platform) for item in sources):
            raise AmendmentArtifactError("A change to edition 1 fixed guarantees requires a verified platform-authored release")
    return genesis


def _verify_legacy_declaration(
    artifact: Mapping, sovereign_document: Mapping, target_did: str, prior_hash: str, new_hash: str,
    identity_directory: Path | None, hosted_identity_evidence_path: Path | None, active_signing_did: str | None,
    declaration_scope: str,
) -> dict:
    from .transition import verify_live_genesis_proof

    proof = artifact.get(LEGACY_FIXED_AMENDMENT_FIELD)
    intent = proof.get("intent") if isinstance(proof, dict) else None
    admission_at = intent.get("admission_at") if isinstance(intent, dict) else None
    expected = {
        "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE,
        "target_did": target_did,
        "sovereign_did": sovereign_document.get("id"),
        "prior_sha256": prior_hash,
        "constitution_sha256": new_hash,
        "admission_at": admission_at,
    }
    if (
        not isinstance(admission_at, str)
        or not isinstance(proof, dict) or set(proof) != {"intent", "genesis_proof"}
        or proof["intent"] != expected
    ):
        raise AmendmentArtifactError(
            f"Changes to {declaration_scope} require a Genesis-verifiable declaration "
            "for this agent and its exact prior and proposed governing digests"
        )
    try:
        when = datetime.fromisoformat(admission_at)
    except ValueError as exc:
        raise AmendmentArtifactError("Edition 1 amendment requires a valid admission time") from exc
    if when.utcoffset() != timezone.utc.utcoffset(when) or when > datetime.now(timezone.utc):
        raise AmendmentArtifactError("Edition 1 amendment admission time must be UTC and not in the future")
    return verify_live_genesis_proof(
        target_did, expected, proof["genesis_proof"], sovereign_document, identity_directory,
        hosted_identity_evidence_path=hosted_identity_evidence_path, active_signing_did=active_signing_did,
    )
