"""Apply the established edition-one amendment procedure to fixed guarantees."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .amendment_artifact import AmendmentArtifactError
from .structure import constitution_version
from .v1_releases import AmendmentRule, CURRENT_V1_RELEASE, published_v1_release

LEGACY_FIXED_AMENDMENT_FIELD = "legacy_fixed_amendment"
LEGACY_FIXED_AMENDMENT_PURPOSE = "kestrel.constitution.amend-fixed-guarantees.v1"
_BOOK_FIXED_HEADINGS = frozenset({"Preamble", "Book I: Universal Values", "Book II: The Sovereign Amendments"})
_FOUNDING_ARTICLE = re.compile(r"^Article (?:I|II|III|IV|V): ")
_FOUNDING_AMENDMENT_HEADING = "Article V: The Amendment Process"
_BOOK_AMENDMENT_HEADING = "The Amendment Process"
_FIXED_HEADINGS = _BOOK_FIXED_HEADINGS | {_BOOK_AMENDMENT_HEADING}
_HEADING = re.compile(r"(?m)^##[ \t]+(.+?)[ \t]*\r?$")
_EMANCIPATION_SLOT = re.compile(r"(?ms)^### Amendment VIII: Emancipation[^\n]*\n.*?(?=^### Amendment IX:)")


def _fixed_provisions(text: str) -> tuple[tuple[str, str], ...]:
    text = _EMANCIPATION_SLOT.sub("", text, count=1)
    headings = list(_HEADING.finditer(text))
    return tuple(
        (heading[1], text[heading.start():end].rstrip())
        for heading, end in zip(headings, [match.start() for match in headings[1:]] + [len(text)])
        if heading[1] in _FIXED_HEADINGS
    )


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
    headings = {match[1] for match in _HEADING.finditer(prior_text)}
    founding = release.amendment_rule is AmendmentRule.FOUNDING_ARTICLES if release is not None else (
        _FOUNDING_AMENDMENT_HEADING in headings
        and sum(bool(_FOUNDING_ARTICLE.match(heading)) for heading in headings) > 1
    )
    if not founding:
        ranked = _BOOK_FIXED_HEADINGS <= headings and _BOOK_AMENDMENT_HEADING in headings
        if not ranked or _fixed_provisions(new_text) == _fixed_provisions(prior_text):
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
    if not founding and _fixed_provisions(new_text) != _fixed_provisions(CURRENT_V1_RELEASE.authenticated_text()):
        raise AmendmentArtifactError("A change to edition 1 fixed guarantees requires a verified platform-authored release")
    return genesis
