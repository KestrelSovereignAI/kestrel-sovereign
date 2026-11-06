"""The Sovereign-signed manifest of material held out of every governed corpus (#3893).

:class:`~kestrel_sovereign.knowledge.corpus.GovernedCorpusPolicy` can hold out
specific saved items, sessions and source occurrences (#3849), but a policy is
the learning consumer's value. A consumer that builds its own policy names no
exclusions, and nothing else supplied any, so held-out material such as a
fidelity exam entered every corpus.

This manifest is where exclusions come from. The host reads it on every
governed corpus build and holds out what it names for the building agent,
whatever the consumer's policy says. It is:

* **Host scope.** One file outside every agent's data directory, and never in
  an agent's saved-item store, which the agent writes: the path in
  ``KESTREL_CORPUS_EXCLUSIONS_PATH``, else ``<host data>/corpus-exclusions.json``.
* **Signed by the Sovereign.** It must verify against the operator-pinned
  trust root (``KESTREL_SOVEREIGN_TRUST_ROOT_PATH``), the root that authorizes
  a constitution reanchor, and that root may not be the agent's own DID.
  Agents and the host can run as one OS user, so file permissions separate
  nothing: an agent can delete or replace the file, but cannot forge one.
* **Per agent.** ``agents`` maps each agent DID to its typed ids
  (``source_occurrence``, ``session``, ``saved_item``), each the full id.
* **Monotonic.** ``sequence`` only moves forward. Each build records the
  highest sequence it has admitted and refuses a lower one, so an older,
  validly signed manifest that excluded less cannot be put back
  (``kestrel_sovereign.storage.corpus_exclusion_watermarks``).

A build fails closed instead of running without exclusions when the manifest is
missing, unreadable or not signed by the trust root; when an id is malformed,
shortened or of an unknown kind; when the sequence moved back; when the agent
has no section; and when the agent's section names a kind no producer records
yet, which would silently match nothing. "No exclusions" is a validly signed
manifest with an empty section for the agent, never an absent file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign._regular_file import NotARegularFileError, read_regular_file
from kestrel_sovereign.constitution.amendment_artifact import verify_detached_signature
from kestrel_sovereign.constitution.trust_root import (
    SovereignTrustRootError,
    load_sovereign_trust_root,
)
from kestrel_sovereign.identity.hybrid_keypair import HybridKeypair, sign_hybrid
from kestrel_sovereign.paths import host_data_dir
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.storage.session_id_column import is_stampable_session_id

from .corpus import CorpusExclusionReceipt, GovernedCorpusUnavailable, HostCorpusExclusions


CORPUS_EXCLUSIONS_TYPE = "kestrel.corpus.exclusions.v1"
CORPUS_EXCLUSIONS_VERSION = 1
CORPUS_EXCLUSIONS_SUBJECT = "governed_corpus_exclusions"
CORPUS_EXCLUSIONS_PATH_ENV = "KESTREL_CORPUS_EXCLUSIONS_PATH"
DEFAULT_CORPUS_EXCLUSIONS_FILENAME = "corpus-exclusions.json"
MAX_CORPUS_EXCLUSIONS_BYTES = 1024 * 1024
#: The largest sequence a 64-bit database integer holds.
MAX_CORPUS_EXCLUSIONS_SEQUENCE = 2**63 - 1

#: Every field the Sovereign signs, so the canonical bytes, the builders and
#: the unknown-field refusal cannot disagree.
_SIGNED_FIELDS = (
    "artifact_type",
    "version",
    "signer",
    "subject",
    "sequence",
    "issued_at",
    "reason",
    "agents",
)
_SIGNATURE_FIELDS = ("signature", "signatures")


class CorpusExclusionKind(str, Enum):
    """What one excluded id names."""

    SOURCE_OCCURRENCE = "source_occurrence"
    SESSION = "session"
    SAVED_ITEM = "saved_item"


#: Kinds a corpus can match today. ``save_fact`` and the strategy-ledger
#: projection name their session (#3875); no producer yet names the saved item
#: a source was read from, so a ``saved_item`` exclusion would match nothing
#: and the material would enter the corpus. A section naming one refuses until
#: a producer records saved items (``SourceReferenceKind.SAVED_ITEM``).
MATCHABLE_EXCLUSION_KINDS = frozenset(
    {CorpusExclusionKind.SOURCE_OCCURRENCE, CorpusExclusionKind.SESSION}
)

#: ``source:<producer version>:<sha256>``: every producer's source id
#: (``save_fact``, the strategy ledger, the legacy-fact migration).
_SOURCE_OCCURRENCE_ID = re.compile(r"source:[a-z0-9][a-z0-9.-]*:[0-9a-f]{64}")
#: A saved item's id is a lowercase UUID4 (``saved_items_store.save_item``).
_SAVED_ITEM_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_UUID_ALPHABET = frozenset("0123456789abcdefABCDEF-")
_AGENT_DID = re.compile(r"did:[a-z0-9]+:[\x21-\x7e]+")


class CorpusExclusionManifestError(GovernedCorpusUnavailable):
    """No governed corpus may be built under the configured exclusion manifest."""


def _refuse(message: str) -> CorpusExclusionManifestError:
    return CorpusExclusionManifestError(f"corpus exclusion manifest refused: {message}")


def validate_exclusion_id(kind: CorpusExclusionKind, value: object) -> str:
    """Return ``value`` when it is one whole id of ``kind``; refuse anything else.

    Ids are compared exactly, so a shortened id, such as the 8-character prefix
    a console shows, would exclude nothing. A session id written in the UUID
    alphabet must therefore be the whole UUID. Other session keys (``sess_*``)
    must be keys the conversation store can file a row under.
    """
    if not isinstance(value, str):
        raise _refuse(f"a {kind.value} id must be a string, not {type(value).__name__}")
    if kind is CorpusExclusionKind.SOURCE_OCCURRENCE:
        valid = bool(_SOURCE_OCCURRENCE_ID.fullmatch(value))
        expected = "source:<producer>:<64 lowercase hex characters>"
    elif kind is CorpusExclusionKind.SAVED_ITEM:
        valid = bool(_SAVED_ITEM_ID.fullmatch(value))
        expected = "a whole lowercase UUID"
    else:
        valid = is_stampable_session_id(value) and (
            not set(value) <= _UUID_ALPHABET or bool(_UUID.fullmatch(value))
        )
        expected = "a whole session id (a UUID session id needs all 36 characters)"
    if not valid:
        raise _refuse(
            f"{kind.value} id {value[:80]!r} is malformed or shortened; expected "
            f"{expected}"
        )
    return value


def canonical_corpus_exclusions_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The bytes a manifest signature covers.

    ``artifact_type`` is among them, so no other Sovereign-signed artifact's
    signature verifies as a manifest's.
    """
    return json.dumps(
        {field: manifest.get(field) for field in _SIGNED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _issued_at(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse("issued_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise _refuse(f"issued_at {value!r} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise _refuse("issued_at must carry a UTC offset")
    return value


def _sections(
    value: object,
) -> dict[str, dict[CorpusExclusionKind, tuple[str, ...]]]:
    if not isinstance(value, Mapping):
        raise _refuse("agents must map each agent DID to its exclusions")
    sections: dict[str, dict[CorpusExclusionKind, tuple[str, ...]]] = {}
    for agent_id, section in value.items():
        if not isinstance(agent_id, str) or not _AGENT_DID.fullmatch(agent_id):
            raise _refuse(f"agent key {agent_id!r} is not an agent DID")
        if not isinstance(section, Mapping):
            raise _refuse(f"the section for {agent_id} must be an object of typed ids")
        typed: dict[CorpusExclusionKind, tuple[str, ...]] = {}
        for kind_name, ids in section.items():
            try:
                kind = CorpusExclusionKind(kind_name)
            except ValueError:
                raise _refuse(
                    f"the section for {agent_id} names unknown id kind {kind_name!r}; "
                    f"expected one of {sorted(item.value for item in CorpusExclusionKind)}"
                ) from None
            if not isinstance(ids, list):
                raise _refuse(f"{agent_id} {kind.value} ids must be a list")
            normalized = tuple(validate_exclusion_id(kind, item) for item in ids)
            if len(set(normalized)) != len(normalized):
                raise _refuse(f"{agent_id} lists a {kind.value} id twice")
            if normalized:
                typed[kind] = tuple(sorted(normalized))
        sections[agent_id] = typed
    return sections


def _validate_signed_fields(
    manifest: Mapping[str, Any],
) -> dict[str, dict[CorpusExclusionKind, tuple[str, ...]]]:
    """Refuse a manifest whose signed fields are not exactly well-formed.

    Shared by the builders and the verifier, so operator tooling cannot sign a
    manifest the host would refuse.
    """
    if manifest.get("artifact_type") != CORPUS_EXCLUSIONS_TYPE:
        raise _refuse(
            f"artifact_type is {manifest.get('artifact_type')!r}, not "
            f"{CORPUS_EXCLUSIONS_TYPE!r}"
        )
    version = manifest.get("version")
    # ``True == 1`` in Python; a JSON ``true`` is not version 1.
    if type(version) is not int or version != CORPUS_EXCLUSIONS_VERSION:
        raise _refuse(f"unsupported version {version!r}")
    if manifest.get("subject") != CORPUS_EXCLUSIONS_SUBJECT:
        raise _refuse(f"subject is not {CORPUS_EXCLUSIONS_SUBJECT!r}")
    signer = manifest.get("signer")
    if not isinstance(signer, str) or not signer.startswith("did:"):
        raise _refuse("it names no signer DID")
    sequence = manifest.get("sequence")
    if type(sequence) is not int or not 1 <= sequence <= MAX_CORPUS_EXCLUSIONS_SEQUENCE:
        raise _refuse(f"sequence {sequence!r} is not a positive 64-bit integer")
    _issued_at(manifest.get("issued_at"))
    if not isinstance(manifest.get("reason"), str):
        raise _refuse("reason must be a string")
    return _sections(manifest.get("agents"))


def _unsigned_manifest(
    *,
    signer_did: str,
    sequence: int,
    agents: Mapping[str, Mapping[str, list[str]]],
    issued_at: str | None,
    reason: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "artifact_type": CORPUS_EXCLUSIONS_TYPE,
        "version": CORPUS_EXCLUSIONS_VERSION,
        "signer": signer_did,
        "subject": CORPUS_EXCLUSIONS_SUBJECT,
        "sequence": sequence,
        "issued_at": issued_at or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        # Lists are signed sorted, so one set of ids has one signed form;
        # anything that is not a list is left for validation to refuse.
        "agents": {
            agent_id: (
                {
                    kind: sorted(ids) if isinstance(ids, list) else ids
                    for kind, ids in section.items()
                }
                if isinstance(section, Mapping)
                else section
            )
            for agent_id, section in agents.items()
        },
    }
    _validate_signed_fields(manifest)
    return manifest


def build_legacy_signed_corpus_exclusions(
    *,
    signer_did: str,
    sequence: int,
    agents: Mapping[str, Mapping[str, list[str]]],
    private_key: ec.EllipticCurvePrivateKey,
    issued_at: str | None = None,
    reason: str = "",
    kid: str = "keys-1",
) -> dict[str, Any]:
    """Build an ECDSA-signed exclusion manifest for operator tooling."""
    manifest = _unsigned_manifest(
        signer_did=signer_did,
        sequence=sequence,
        agents=agents,
        issued_at=issued_at,
        reason=reason,
    )
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    signature = suite.sign(canonical_corpus_exclusions_bytes(manifest), private_key)
    manifest["signature"] = {
        "alg": ALG_ECDSA_SECP256K1_SHA256,
        "kid": kid,
        "sig": signature.hex(),
    }
    return manifest


def build_hybrid_signed_corpus_exclusions(
    *,
    signer_did: str,
    sequence: int,
    agents: Mapping[str, Mapping[str, list[str]]],
    keypair: HybridKeypair,
    issued_at: str | None = None,
    reason: str = "",
    classical_kid: str = "key-1",
    pq_kid: str = "key-2",
) -> dict[str, Any]:
    """Build a hybrid-signed exclusion manifest for operator tooling."""
    manifest = _unsigned_manifest(
        signer_did=signer_did,
        sequence=sequence,
        agents=agents,
        issued_at=issued_at,
        reason=reason,
    )
    manifest["signatures"] = sign_hybrid(
        canonical_corpus_exclusions_bytes(manifest),
        keypair,
        classical_kid=classical_kid,
        pq_kid=pq_kid,
    )
    return manifest


@dataclass(frozen=True)
class VerifiedCorpusExclusionManifest:
    """A manifest whose signature verified against the pinned trust root."""

    path: str
    signer: str
    verification: str
    sequence: int
    issued_at: str
    reason: str
    manifest_sha256: str
    sections: Mapping[str, Mapping[CorpusExclusionKind, tuple[str, ...]]]

    @property
    def receipt(self) -> CorpusExclusionReceipt:
        return CorpusExclusionReceipt(self.manifest_sha256, self.sequence, self.issued_at)

    def unmatchable_kinds(self, agent_id: str) -> tuple[CorpusExclusionKind, ...]:
        """The kinds ``agent_id``'s section names that no corpus can match yet."""
        section = self.sections.get(agent_id, {})
        return tuple(
            kind for kind in CorpusExclusionKind
            if section.get(kind) and kind not in MATCHABLE_EXCLUSION_KINDS
        )

    def exclusions_for(self, agent_id: str) -> HostCorpusExclusions:
        """What every corpus built for ``agent_id`` holds out under this manifest."""
        section = self.sections.get(agent_id)
        if section is None:
            raise _refuse(
                f"{self.path} (sequence {self.sequence}) has no section for "
                f"{agent_id}. A corpus with no exclusions needs an explicit, "
                "signed empty section for the agent."
            )
        unmatchable = self.unmatchable_kinds(agent_id)
        if unmatchable:
            raise _refuse(
                f"the section for {agent_id} names "
                f"{', '.join(kind.value for kind in unmatchable)} ids, which no "
                "producer records on its sources yet, so they would match "
                "nothing and the material would enter the corpus. Exclude that "
                "material by its source_occurrence or session ids instead."
            )
        return HostCorpusExclusions(
            receipt=self.receipt,
            saved_item_ids=section.get(CorpusExclusionKind.SAVED_ITEM, ()),
            session_ids=section.get(CorpusExclusionKind.SESSION, ()),
            source_occurrence_ids=section.get(
                CorpusExclusionKind.SOURCE_OCCURRENCE, ()
            ),
        )


def verify_corpus_exclusions(
    manifest: Mapping[str, Any],
    *,
    trusted_did_document: Mapping[str, Any],
) -> tuple[str, str, dict[str, dict[CorpusExclusionKind, tuple[str, ...]]]]:
    """Verify a parsed manifest against the pinned Sovereign trust root.

    Returns ``(signer, verification, sections)``. Unknown top-level fields are
    refused rather than ignored: an unsigned field beside signed ones is an
    ambiguity no reader should have to resolve.
    """
    if not isinstance(manifest, Mapping):
        raise _refuse("it must be a JSON object")
    unknown = sorted(set(manifest) - set(_SIGNED_FIELDS) - set(_SIGNATURE_FIELDS))
    if unknown:
        raise _refuse(f"it carries unsigned field(s) {unknown}")
    missing = sorted(set(_SIGNED_FIELDS) - set(manifest))
    if missing:
        raise _refuse(f"it is missing {missing}")
    if len([field for field in _SIGNATURE_FIELDS if field in manifest]) != 1:
        raise _refuse(
            "it must carry exactly one of 'signature' (legacy) or 'signatures' (hybrid)"
        )
    sections = _validate_signed_fields(manifest)
    signer = manifest["signer"]
    trusted_did = str(trusted_did_document.get("id") or "")
    if signer != trusted_did:
        raise _refuse(
            f"signer {signer!r} is not the trusted Sovereign DID {trusted_did!r}"
        )
    ok, verification = verify_detached_signature(
        canonical_corpus_exclusions_bytes(manifest),
        manifest,
        trusted_did_document=trusted_did_document,
    )
    if not ok:
        raise _refuse(f"signature verification failed: {verification}")
    return signer, verification, sections


def configured_corpus_exclusions_path(
    *, environ: Mapping[str, str] | None = None
) -> Path:
    """The one manifest file a host reads: the configured path, else the default.

    A configured path must be absolute, so every process of a host reads the
    same file whatever its working directory. Set but empty is refused: a
    blanked-out path is not a request for the default.
    """
    env = os.environ if environ is None else environ
    configured = env.get(CORPUS_EXCLUSIONS_PATH_ENV)
    if configured is None:
        return host_data_dir(environ=env) / DEFAULT_CORPUS_EXCLUSIONS_FILENAME
    if not configured.strip():
        raise _refuse(
            f"{CORPUS_EXCLUSIONS_PATH_ENV} is set but empty; name the manifest, "
            "or unset it to use "
            f"{host_data_dir(environ=env) / DEFAULT_CORPUS_EXCLUSIONS_FILENAME}"
        )
    path = Path(configured.strip()).expanduser()
    if not path.is_absolute():
        raise _refuse(f"{CORPUS_EXCLUSIONS_PATH_ENV}={configured!r} is not an absolute path")
    return path


def _refuse_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _refuse("a JSON object repeats a key")
    return dict(pairs)


def load_corpus_exclusions(
    *,
    environ: Mapping[str, str] | None = None,
    trust_root_path: str | os.PathLike[str] | None = None,
    agent_dids: set[str] | frozenset[str] = frozenset(),
) -> VerifiedCorpusExclusionManifest:
    """Read and verify the host's exclusion manifest, refusing anything less.

    ``trust_root_path`` and ``environ`` select the Sovereign trust root as
    :func:`~kestrel_sovereign.constitution.trust_root.load_sovereign_trust_root`
    does; ``agent_dids`` are DIDs the root may not be.

    Synchronous file I/O: an event-loop caller runs it in a worker thread
    (``AsyncStorage`` does, within the build's time budget). Neither file can
    block it: anything but a regular file is refused without being read.
    """
    path = configured_corpus_exclusions_path(environ=environ)
    try:
        trusted = load_sovereign_trust_root(
            explicit_path=trust_root_path, environ=environ, agent_dids=agent_dids
        )
    except SovereignTrustRootError as error:
        raise _refuse(f"no Sovereign trust root verifies it: {error}") from error
    try:
        raw = read_regular_file(path, max_bytes=MAX_CORPUS_EXCLUSIONS_BYTES)
    except FileNotFoundError as error:
        raise _refuse(
            f"no manifest at {path}. Every governed corpus build needs a "
            "Sovereign-signed manifest; one that excludes nothing has an empty "
            "section for the agent."
        ) from error
    except NotARegularFileError as error:
        # A FIFO put in its place would otherwise block the read until
        # something wrote to it.
        raise _refuse(f"{path} is not a regular file") from error
    except OSError as error:
        raise _refuse(f"cannot read {path}: {error}") from error
    if len(raw) > MAX_CORPUS_EXCLUSIONS_BYTES:
        raise _refuse(f"{path} exceeds {MAX_CORPUS_EXCLUSIONS_BYTES} bytes")
    try:
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_refuse_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _refuse(f"{path} is not valid JSON: {error}") from error
    signer, verification, sections = verify_corpus_exclusions(
        manifest, trusted_did_document=trusted
    )
    return VerifiedCorpusExclusionManifest(
        path=str(path),
        signer=signer,
        verification=verification,
        sequence=manifest["sequence"],
        issued_at=manifest["issued_at"],
        reason=manifest["reason"],
        manifest_sha256=hashlib.sha256(
            canonical_corpus_exclusions_bytes(manifest)
        ).hexdigest(),
        sections=MappingProxyType(
            {agent_id: MappingProxyType(section) for agent_id, section in sections.items()}
        ),
    )


__all__ = [
    "CORPUS_EXCLUSIONS_PATH_ENV",
    "CORPUS_EXCLUSIONS_SUBJECT",
    "CORPUS_EXCLUSIONS_TYPE",
    "CORPUS_EXCLUSIONS_VERSION",
    "CorpusExclusionKind",
    "CorpusExclusionManifestError",
    "DEFAULT_CORPUS_EXCLUSIONS_FILENAME",
    "MATCHABLE_EXCLUSION_KINDS",
    "VerifiedCorpusExclusionManifest",
    "build_hybrid_signed_corpus_exclusions",
    "build_legacy_signed_corpus_exclusions",
    "canonical_corpus_exclusions_bytes",
    "configured_corpus_exclusions_path",
    "load_corpus_exclusions",
    "validate_exclusion_id",
    "verify_corpus_exclusions",
]
