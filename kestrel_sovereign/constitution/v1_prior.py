"""Identify exact edition-one custody and independently signed role revisions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterator, Mapping

from .amendment_artifact import AmendmentArtifactError, verify_reanchor_artifact
from .emancipation import EmancipationContract, apply_emancipation, matches_recorded_emancipation_contract
from .reanchor_receipt import CONSTITUTION_REANCHOR_KEY, is_constitution_hash

from .v1_releases import V1_RELEASES, published_v1_release

_ROLE = re.compile(r"(?m)^## Book IV: Agent Identity[ \t]*\r?$")
_NEXT_SECTION = re.compile(r"(?m)^##[ \t]+")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def known_published_v1(digest: str) -> bool:
    return published_v1_release(digest) is not None


def _supported_texts(contract: EmancipationContract | None) -> Iterator[str]:
    for release in V1_RELEASES:
        if contract is None or not contract.enabled:
            yield release.authenticated_text()
        elif release.supports_activation:
            yield apply_emancipation(release.authenticated_text(), contract)


def recover_published_v1(
    digest: str, anchored_contract: EmancipationContract | None = None,
) -> str | None:
    """Resolve exactly attested v1 bytes, including an authenticated active render."""
    if anchored_contract is None or not anchored_contract.enabled:
        release = published_v1_release(digest)
        return release.authenticated_text() if release is not None else None
    return next((text for text in _supported_texts(anchored_contract) if _digest(text) == digest), None)


def recover_missing_v1_history(
    digest: str, governing_text: str | None, anchored_contract: EmancipationContract | None,
) -> str | None:
    """Recover missing v1 history only with exact bytes and any active terms preserved."""
    published = recover_published_v1(digest)
    if published is not None:
        return published
    if (
        anchored_contract is None or not anchored_contract.enabled or governing_text is None
        or not matches_recorded_emancipation_contract(governing_text, anchored_contract)
    ):
        return None
    return recover_published_v1(digest, anchored_contract)


def _role_frame(text: str) -> str | None:
    headings = list(_ROLE.finditer(text))
    if len(headings) != 1:
        return None
    heading = headings[0]
    following = _NEXT_SECTION.search(text, heading.end())
    if following is None or "\x00" in text:
        return None
    return _digest(text[:heading.end()] + "\x00" + text[following.start():])


def _exact_v1(prior_hash: str, contract: EmancipationContract | None) -> bool:
    return recover_published_v1(prior_hash, contract) is not None


def _in_supported_role_frame(text: str, contract: EmancipationContract | None) -> bool:
    frame = _role_frame(text)
    return frame is not None and any(frame == _role_frame(source) for source in _supported_texts(contract))


async def read_prior_role_artifact(
    db, properties: Mapping, *, prior_hash: str | None, prior_text: str | None,
    anchored_contract: EmancipationContract | None,
) -> bytes | None:
    """Read the prior scoped admission artifact through the native receipt."""
    if prior_hash is None or prior_text is None or _exact_v1(prior_hash, anchored_contract):
        return None
    if not _in_supported_role_frame(prior_text, anchored_contract):
        return None
    current = properties.get(CONSTITUTION_REANCHOR_KEY)
    if not isinstance(current, Mapping) or current.get("new_hash") != prior_hash:
        return None
    artifact_hash = current.get("signed_artifact_hash")
    if not is_constitution_hash(artifact_hash):
        return None
    from .anchored_bytes import read_anchored_constitution

    text, _ = await read_anchored_constitution(db, artifact_hash)
    return text.encode("utf-8") if text is not None else None


def verify_supported_v1_prior(
    prior_hash: str, contract: EmancipationContract | None, *,
    prior_text: str | None = None, prior_role_artifact_bytes: bytes | None = None,
    trusted_did_document: Mapping | None = None,
) -> dict[str, str]:
    """Require a published digest or a signed prior Book IV-only admission."""
    if _exact_v1(prior_hash, contract):
        return {}
    if (
        prior_text is not None and _digest(prior_text) == prior_hash
        and _in_supported_role_frame(prior_text, contract)
        and prior_role_artifact_bytes is not None and trusted_did_document is not None
    ):
        try:
            artifact = json.loads(prior_role_artifact_bytes)
            verified = isinstance(artifact, dict) and verify_reanchor_artifact(
                artifact, trusted_did_document=trusted_did_document, expected_constitution_sha256=prior_hash,
            ).ok
        except (TypeError, ValueError):
            verified = False
        if verified:
            return {"prior_role_artifact_sha256": hashlib.sha256(prior_role_artifact_bytes).hexdigest()}
    raise AmendmentArtifactError(
        "Cannot establish the prior edition 1 adoption authority or preserve its promises. "
        "Provide exact governing bytes from a supported public platform revision, or unchanged "
        "published provisions outside Book IV and the independently verifiable signed prior role admission."
    )
