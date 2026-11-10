"""A test Sovereign that signs the host's corpus exclusion manifest (#3893).

Every governed corpus build refuses without a Sovereign-signed exclusion
manifest naming the building agent. :class:`ExclusionAuthority` pins a fresh
Sovereign trust root and a manifest path in the environment (through
``monkeypatch``) and publishes manifests signed by that root, each with the
next sequence. A test that builds a corpus calls :meth:`admit` for its agent,
which publishes an empty section: the agent's corpus then excludes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from kestrel_sovereign.constitution.amendment_artifact import (
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.trust_root import SOVEREIGN_TRUST_ROOT_ENV
from kestrel_sovereign.knowledge.corpus_exclusions import (
    CORPUS_EXCLUSIONS_PATH_ENV,
    build_legacy_signed_corpus_exclusions,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite

SOVEREIGN_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000003893"


class ExclusionAuthority:
    """The operator-pinned Sovereign of one test host, and its manifest file."""

    def __init__(self, root: Path, monkeypatch) -> None:
        root.mkdir(parents=True, exist_ok=True)
        keypair = Secp256k1Suite().generate_keypair()
        self.private_key = keypair.private_key
        self.did_document = did_document_from_legacy_public_key(
            SOVEREIGN_DID, keypair.public_key
        )
        self.trust_root_path = root / "sovereign-root.did.json"
        self.trust_root_path.write_text(json.dumps(self.did_document))
        self.manifest_path = root / "corpus-exclusions.json"
        self.sections: dict[str, dict[str, list[str]]] = {}
        self.sequence = 0
        monkeypatch.setenv(SOVEREIGN_TRUST_ROOT_ENV, str(self.trust_root_path))
        monkeypatch.setenv(CORPUS_EXCLUSIONS_PATH_ENV, str(self.manifest_path))

    def sign(
        self,
        sections: Mapping[str, Mapping[str, list[str]]],
        *,
        sequence: int,
        reason: str = "test exclusions",
        issued_at: str | None = None,
    ) -> dict[str, Any]:
        return build_legacy_signed_corpus_exclusions(
            signer_did=SOVEREIGN_DID,
            sequence=sequence,
            agents=sections,
            private_key=self.private_key,
            issued_at=issued_at,
            reason=reason,
        )

    def write(self, manifest: Mapping[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest))

    def exclude(self, agent_id: str, **ids: list[str]) -> dict[str, Any]:
        """Set ``agent_id``'s section to ``ids`` and publish the next sequence."""
        self.sections[agent_id] = {kind: list(values) for kind, values in ids.items()}
        self.sequence += 1
        manifest = self.sign(self.sections, sequence=self.sequence)
        self.write(manifest)
        return manifest

    def admit(self, agent_id: str) -> dict[str, Any]:
        """Publish an empty section for ``agent_id``: its corpus excludes nothing."""
        return self.exclude(agent_id)
