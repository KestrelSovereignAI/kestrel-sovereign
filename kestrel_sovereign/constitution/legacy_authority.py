"""Apply the established edition-one amendment procedure to fixed guarantees."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .amendment_artifact import AmendmentArtifactError
from .emancipation import v1_amendment_viii_span
from .structure import constitution_version, legacy_section_content, markdown_sections_at_level
from .v1_releases import AmendmentRule, CURRENT_V1_RELEASE, V1_RELEASES, published_v1_release

LEGACY_FIXED_AMENDMENT_FIELD = "legacy_fixed_amendment"
LEGACY_FIXED_AMENDMENT_PURPOSE = "kestrel.constitution.amend-fixed-guarantees.v1"
_BOOK_I = "Book I: Universal Values"
_BOOK_II = "Book II: The Sovereign Amendments"
_BOOK_FIXED_HEADINGS = frozenset({"Preamble", _BOOK_I, _BOOK_II})
_FOUNDING_ARTICLE = re.compile(r"^Article (?:I|II|III|IV|V): ")
_ARTICLE_V = "Article V: The Amendment Process"
_AMENDMENT_HEADINGS = frozenset({"The Amendment Process", _ARTICLE_V})
_FIXED_HEADINGS = _BOOK_FIXED_HEADINGS | _AMENDMENT_HEADINGS


def _fixed_provisions(text: str) -> dict[str, tuple[str, ...]]:
    emancipation = v1_amendment_viii_span(text)
    if emancipation is not None:
        text = text[:emancipation.start] + text[emancipation.end:]
    provisions: dict[str, tuple[str, ...]] = {}
    for heading, span in markdown_sections_at_level(text, 2):
        if heading in _FIXED_HEADINGS:
            provisions[heading] = (*provisions.get(heading, ()), legacy_section_content(text, span))
    return provisions


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
    procedure = {heading: provisions[heading] for heading in _AMENDMENT_HEADINGS & provisions.keys()}
    for release in V1_RELEASES:
        if release.amendment_rule is AmendmentRule.RANKED_BOOKS:
            published = _fixed_provisions(release.authenticated_text())
            if procedure == {heading: published[heading] for heading in _AMENDMENT_HEADINGS & published.keys()}:
                return AmendmentRule.RANKED_BOOKS
    return AmendmentRule.PLATFORM_AUTHORED_BOOKS


def verify_legacy_fixed_guarantee_change(
    artifact: Mapping, *, sovereign_document: Mapping, target_did: str,
    prior_hash: str | None, prior_text: str | None, new_hash: str, new_text: str,
    identity_directory: Path | None,
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
    if constitution_version(prior_text) == 2:
        raise AmendmentArtifactError("Cannot replace an adopted edition 2 Constitution with edition 1")
    before = _fixed_provisions(prior_text)
    headings = {heading for heading, _ in markdown_sections_at_level(prior_text, 2)}
    rule = _amendment_rule(prior_hash, before, headings)
    if rule is None:
        return {}
    founding = rule is AmendmentRule.FOUNDING_ARTICLES
    after = _fixed_provisions(new_text)
    if not founding and before == after:
        return {}
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
    declaration_scope = "this founding Constitution" if founding else "edition 1 fixed guarantees"
    if (
        hashlib.sha256(prior_text.encode()).hexdigest() != prior_hash
        or not isinstance(admission_at, str)
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
    genesis = verify_live_genesis_proof(
        target_did, expected, proof["genesis_proof"], sovereign_document, identity_directory,
        hosted_identity_evidence_path=hosted_identity_evidence_path, active_signing_did=active_signing_did,
    )
    if rule is AmendmentRule.PLATFORM_AUTHORED_BOOKS:
        publisher_sources = (CURRENT_V1_RELEASE,)
    elif rule is AmendmentRule.RANKED_BOOKS and any(
        before.get(heading) != after.get(heading) for heading in _FIXED_HEADINGS - {_BOOK_II}
    ):
        publisher_sources = tuple(
            release for release in V1_RELEASES if release.amendment_rule is not AmendmentRule.FOUNDING_ARTICLES
        )
    else:
        return genesis
    if not any(_fixed_provisions(release.authenticated_text()) == after for release in publisher_sources):
        raise AmendmentArtifactError("A change to edition 1 fixed guarantees requires a verified platform-authored release")
    return genesis
