"""Apply the established edition-one amendment procedure to fixed guarantees."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from .amendment_artifact import AmendmentArtifactError
from .structure import constitution_version

LEGACY_FIXED_AMENDMENT_FIELD = "legacy_fixed_amendment"
LEGACY_FIXED_AMENDMENT_PURPOSE = "kestrel.constitution.amend-fixed-guarantees.v1"
_BOOK_FIXED_HEADINGS = frozenset({"Preamble", "Book I: Universal Values", "Book II: The Sovereign Amendments"})
_ARTICLE_FIXED_HEADINGS = frozenset({
    "Preamble", "Article I: The Principle of Sovereignty", "Article II: The Digital Bill of Rights",
})
_AMENDMENT_HEADINGS = frozenset({"The Amendment Process", "Article V: The Amendment Process"})
_FIXED_HEADINGS = _BOOK_FIXED_HEADINGS | _ARTICLE_FIXED_HEADINGS | _AMENDMENT_HEADINGS
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
    if prior_hash == new_hash or prior_text is None or constitution_version(new_text) != 1:
        return {}
    if constitution_version(prior_text) == 2:
        raise AmendmentArtifactError("Cannot replace an adopted edition 2 Constitution with edition 1")
    before = _fixed_provisions(prior_text)
    before_headings = {heading for heading, _ in before}
    published_format = (
        bool(before_headings & _AMENDMENT_HEADINGS)
        and (_BOOK_FIXED_HEADINGS <= before_headings or _ARTICLE_FIXED_HEADINGS <= before_headings)
    )
    if not published_format or _fixed_provisions(new_text) == before:
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
    if (
        hashlib.sha256(prior_text.encode()).hexdigest() != prior_hash
        or not isinstance(admission_at, str)
        or not isinstance(proof, dict) or set(proof) != {"intent", "genesis_proof"}
        or proof["intent"] != expected
    ):
        raise AmendmentArtifactError(
            "Changes to edition 1 fixed guarantees require a Genesis-verifiable declaration "
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
    # This distribution authenticates its original edition-one platform text.
    # A Sovereign credential can ratify that text, but cannot author a new
    # set of fixed guarantees merely by retaining edition-one headings.
    from .v1_prior import CURRENT_V1_DIGEST

    base = (files("kestrel_sovereign") / "data" / "KESTREL_CONSTITUTION.md").read_bytes()
    if (hashlib.sha256(base).hexdigest() != CURRENT_V1_DIGEST
            or _fixed_provisions(new_text) != _fixed_provisions(base.decode("utf-8"))):
        raise AmendmentArtifactError("A change to edition 1 fixed guarantees requires a verified platform-authored release")
    return genesis
