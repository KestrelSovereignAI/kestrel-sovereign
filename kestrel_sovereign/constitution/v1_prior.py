"""Identify exact edition-one custody and independently signed role revisions."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re
from typing import Mapping

from .amendment_artifact import AmendmentArtifactError, verify_reanchor_artifact
from .emancipation import EmancipationContract, apply_emancipation, known_historical_platform_promise
from .reanchor_receipt import CONSTITUTION_REANCHOR_KEY, is_constitution_hash

CURRENT_V1_DIGEST = "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467"

# Exact packaged file digest to hash of that file with only the Book IV
# body removed. Provenance in order: 84f4591ab, f0672b394, a292b641.
_V1_RELEASES = {
    "46687f754bc008f4022599fb481ebf9abf0fbf1e8e205472ac8efba95628eb9c": "1aa93b5de3756f88fd6fecf911bbf27af346c244f612069c67213f709bf89b48",
    "08aec7530f912ad2ae154a3687387bd1a73b0489a3532d7f67c99d635372d63f": "b687ea2c0e4cd28812d76ed831319ba03d696de17dda5e5b7e268f4055996ec6",
    CURRENT_V1_DIGEST: "41f746aad43846ac3ff3d7a58573eb294a6e4263ca0d146e3902379e7de52c2d",
}
_ROLE = re.compile(r"(?m)^## Book IV: Agent Identity[ \t]*\r?$")
_NEXT_SECTION = re.compile(r"(?m)^##[ \t]+")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def known_published_v1(digest: str) -> bool:
    return digest in _V1_RELEASES or known_historical_platform_promise(digest)


def witnessed_v1_render(contract: EmancipationContract) -> str:
    try:
        base = (files("kestrel_sovereign") / "data" / "KESTREL_CONSTITUTION.md").read_bytes()
    except OSError as exc:
        raise AmendmentArtifactError("Cannot read the immutable edition 1 source") from exc
    if hashlib.sha256(base).hexdigest() != CURRENT_V1_DIGEST:
        raise AmendmentArtifactError("Installed edition 1 source differs from the authenticated published bytes")
    return apply_emancipation(base.decode("utf-8"), contract)


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
    if contract is None or not contract.enabled:
        return known_published_v1(prior_hash)
    return _digest(witnessed_v1_render(contract)) == prior_hash


def _in_supported_role_frame(text: str, contract: EmancipationContract | None) -> bool:
    frame = _role_frame(text)
    if frame is None:
        return False
    if contract is not None and contract.enabled:
        return frame == _role_frame(witnessed_v1_render(contract))
    return frame in _V1_RELEASES.values()


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
