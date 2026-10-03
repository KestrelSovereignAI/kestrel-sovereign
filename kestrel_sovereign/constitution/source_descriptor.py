"""Sovereign-signed governing-constitution source descriptors (#2553).

Which bytes govern an agent is itself a governance decision. Until #2553 the
answer was fixed: the packaged ``config.CONSTITUTION_PATH``, with every other
source refused (#2463). A *source descriptor* lets the Sovereign choose a
different governing source without handing that choice to the database the
choice protects.

A descriptor is a detached JSON artifact of type
``kestrel.constitution.source.v1``. Its signed fields bind:

* ``source_kind`` — ``package`` (the installed package's constitution) or
  ``external`` (an operator file outside the package);
* ``source_path`` — the absolute path of an ``external`` source, and ``null``
  for ``package``, whose location is the installation's own;
* ``content_sha256`` — the SHA-256 of the source's raw bytes, *before* any
  Amendment VIII rendering. A source that no longer hashes to this digest is an
  integrity failure, for either kind.

Authority comes from two places outside the agent's graph database, and from
nowhere else:

1. **Selection.** The descriptor file is named only by operator configuration:
   an explicit path (``KestrelAgent``, inception, the offline CLI, a
   multi-agent entry) or ``KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH``. No
   graph property, receipt, or anchored blob selects a source. A database
   writer who flips a recorded kind from ``package`` to ``external`` changes
   nothing that the resolver reads, so package-drift enforcement still holds.
2. **Verification.** The signature must verify against the operator-pinned
   Sovereign trust root from
   :func:`kestrel_sovereign.constitution.trust_root.load_sovereign_trust_root`,
   the same resolver live and offline reanchor use (#2499).

Configured-but-unusable is never "use the package instead": a descriptor that
is missing, unreadable, ambiguous, malformed, unsigned, signed by anyone but
the trust root, or whose source drifted raises :class:`ConstitutionSourceError`
so every caller fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign.identity.hybrid_keypair import HybridKeypair, sign_hybrid
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)

from .amendment_artifact import verify_detached_signature


SOURCE_DESCRIPTOR_TYPE = "kestrel.constitution.source.v1"
SOURCE_DESCRIPTOR_VERSION = 1
SOURCE_DESCRIPTOR_SUBJECT = "constitution_source"
SOURCE_KIND_PACKAGE = "package"
SOURCE_KIND_EXTERNAL = "external"
SOURCE_KINDS = frozenset({SOURCE_KIND_PACKAGE, SOURCE_KIND_EXTERNAL})
CONSTITUTION_SOURCE_DESCRIPTOR_ENV = "KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH"
MAX_SOURCE_DESCRIPTOR_BYTES = 64 * 1024

#: Every field the Sovereign signs, in one place so the canonical bytes, the
#: builders, and the unknown-field refusal cannot disagree.
_SIGNED_FIELDS = (
    "artifact_type",
    "version",
    "signer",
    "subject",
    "source_kind",
    "source_path",
    "content_sha256",
    "created_at",
    "reason",
)
_SIGNATURE_FIELDS = ("signature", "signatures")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ConstitutionSourceError(ValueError):
    """The configured governing-constitution source cannot be trusted."""


@dataclass(frozen=True)
class VerifiedSourceDescriptor:
    """A descriptor whose signature verified against the pinned trust root."""

    source_kind: str
    source_path: Optional[str]
    content_sha256: str
    signer: str
    verification: str
    descriptor_path: str
    descriptor_sha256: str


def canonical_source_descriptor_bytes(descriptor: Mapping[str, Any]) -> bytes:
    """Return the stable byte payload a source-descriptor signature covers.

    ``artifact_type`` is part of the payload, so a reanchor-artifact signature
    can never verify as a descriptor signature, or the reverse.
    """
    signed_fields = {field: descriptor.get(field) for field in _SIGNED_FIELDS}
    signed_fields["reason"] = descriptor.get("reason", "")
    return json.dumps(
        signed_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _validate_signed_fields(descriptor: Mapping[str, Any]) -> None:
    """Refuse any descriptor whose signed fields are not exactly well-formed.

    Shared by the builders and the verifier, so operator tooling cannot sign a
    descriptor that the runtime would later refuse.
    """
    if descriptor.get("artifact_type") != SOURCE_DESCRIPTOR_TYPE:
        raise ConstitutionSourceError(
            f"unsupported artifact_type {descriptor.get('artifact_type')!r}; "
            f"expected {SOURCE_DESCRIPTOR_TYPE!r}"
        )
    version = descriptor.get("version")
    # ``True == 1`` in Python; a JSON ``true`` is not version 1.
    if type(version) is not int or version != SOURCE_DESCRIPTOR_VERSION:
        raise ConstitutionSourceError(
            f"unsupported source descriptor version {version!r}"
        )
    if descriptor.get("subject") != SOURCE_DESCRIPTOR_SUBJECT:
        raise ConstitutionSourceError(
            f"descriptor subject is not {SOURCE_DESCRIPTOR_SUBJECT!r}"
        )
    signer = descriptor.get("signer")
    if not isinstance(signer, str) or not signer.startswith("did:"):
        raise ConstitutionSourceError("descriptor has no signer DID")

    source_kind = descriptor.get("source_kind")
    # Type first: a JSON list or object is unhashable, and a set-membership
    # test on it raises TypeError, which no caller handles as an untrusted
    # descriptor.
    if not isinstance(source_kind, str) or source_kind not in SOURCE_KINDS:
        raise ConstitutionSourceError(
            f"unsupported source_kind {source_kind!r}; expected one of "
            f"{sorted(SOURCE_KINDS)}"
        )
    source_path = descriptor.get("source_path")
    if source_kind == SOURCE_KIND_PACKAGE:
        if source_path is not None:
            raise ConstitutionSourceError(
                "a package source descriptor must not name a source_path: the "
                "package source is the installed package's own constitution"
            )
    elif (
        not isinstance(source_path, str)
        or not source_path
        or "\x00" in source_path
        or not os.path.isabs(source_path)
    ):
        raise ConstitutionSourceError(
            "an external source descriptor must name an absolute source_path"
        )

    content_sha256 = descriptor.get("content_sha256")
    if not isinstance(content_sha256, str) or not _SHA256_HEX.fullmatch(
        content_sha256
    ):
        raise ConstitutionSourceError(
            "descriptor content_sha256 must be 64 lowercase hex characters"
        )
    if not isinstance(descriptor.get("created_at"), str):
        raise ConstitutionSourceError("descriptor has no created_at timestamp")
    if not isinstance(descriptor.get("reason", ""), str):
        raise ConstitutionSourceError("descriptor reason must be a string")


def _unsigned_descriptor(
    *,
    signer_did: str,
    source_kind: str,
    content_sha256: str,
    source_path: Optional[str],
    created_at: Optional[str],
    reason: str,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "artifact_type": SOURCE_DESCRIPTOR_TYPE,
        "version": SOURCE_DESCRIPTOR_VERSION,
        "signer": signer_did,
        "subject": SOURCE_DESCRIPTOR_SUBJECT,
        "source_kind": source_kind,
        "source_path": source_path,
        "content_sha256": content_sha256,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _validate_signed_fields(descriptor)
    return descriptor


def build_legacy_signed_source_descriptor(
    *,
    signer_did: str,
    source_kind: str,
    content_sha256: str,
    private_key: ec.EllipticCurvePrivateKey,
    source_path: Optional[str] = None,
    created_at: Optional[str] = None,
    reason: str = "",
    kid: str = "keys-1",
) -> dict[str, Any]:
    """Build an ECDSA-signed source descriptor for operator tooling."""
    descriptor = _unsigned_descriptor(
        signer_did=signer_did,
        source_kind=source_kind,
        content_sha256=content_sha256,
        source_path=source_path,
        created_at=created_at,
        reason=reason,
    )
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    sig = suite.sign(canonical_source_descriptor_bytes(descriptor), private_key)
    descriptor["signature"] = {
        "alg": ALG_ECDSA_SECP256K1_SHA256,
        "kid": kid,
        "sig": sig.hex(),
    }
    return descriptor


def build_hybrid_signed_source_descriptor(
    *,
    signer_did: str,
    source_kind: str,
    content_sha256: str,
    keypair: HybridKeypair,
    source_path: Optional[str] = None,
    created_at: Optional[str] = None,
    reason: str = "",
    classical_kid: str = "key-1",
    pq_kid: str = "key-2",
) -> dict[str, Any]:
    """Build a hybrid-signed source descriptor for operator tooling."""
    descriptor = _unsigned_descriptor(
        signer_did=signer_did,
        source_kind=source_kind,
        content_sha256=content_sha256,
        source_path=source_path,
        created_at=created_at,
        reason=reason,
    )
    descriptor["signatures"] = sign_hybrid(
        canonical_source_descriptor_bytes(descriptor),
        keypair,
        classical_kid=classical_kid,
        pq_kid=pq_kid,
    )
    return descriptor


def verify_source_descriptor(
    descriptor: Mapping[str, Any],
    *,
    trusted_did_document: Mapping[str, Any],
) -> tuple[str, str]:
    """Verify a descriptor against the pinned Sovereign trust root.

    Returns ``(signer, verification_reason)``. Raises
    :class:`ConstitutionSourceError` for anything short of an exactly
    well-formed descriptor signed by the trusted DID. Unknown top-level fields
    are refused rather than ignored: an unsigned field next to signed ones is
    an ambiguity, and no reader should ever have to decide whether to honor it.
    """
    if not isinstance(descriptor, Mapping):
        raise ConstitutionSourceError("source descriptor must be a JSON object")
    unknown = sorted(
        set(descriptor) - set(_SIGNED_FIELDS) - set(_SIGNATURE_FIELDS)
    )
    if unknown:
        raise ConstitutionSourceError(
            f"source descriptor carries unsigned field(s) {unknown}"
        )
    present_signatures = [f for f in _SIGNATURE_FIELDS if f in descriptor]
    if len(present_signatures) != 1:
        raise ConstitutionSourceError(
            "source descriptor must carry exactly one of 'signature' "
            "(legacy) or 'signatures' (hybrid)"
        )
    _validate_signed_fields(descriptor)

    signer = descriptor["signer"]
    trusted_did = str(trusted_did_document.get("id") or "")
    if signer != trusted_did:
        raise ConstitutionSourceError(
            f"descriptor signer {signer!r} is not the trusted Sovereign DID "
            f"{trusted_did!r}"
        )
    ok, reason = verify_detached_signature(
        canonical_source_descriptor_bytes(descriptor),
        descriptor,
        trusted_did_document=trusted_did_document,
    )
    if not ok:
        raise ConstitutionSourceError(
            f"source descriptor signature verification failed: {reason}"
        )
    return signer, reason


def configured_source_descriptor_path(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Optional[Path]:
    """Return the one configured descriptor file, or None when none is set.

    Mirrors :func:`load_sovereign_trust_root`: an explicit path and the
    environment variable may both be set only when they resolve to the same
    file. A configured path that does not exist is an error, never "no
    descriptor" — silently reverting to the package source would let deleting
    a file change which constitution governs.
    """
    env = os.environ if environ is None else environ
    configured: list[tuple[str, Path]] = []
    if explicit_path is not None and str(explicit_path).strip():
        configured.append(("explicit source-descriptor path", Path(explicit_path)))
    env_path = env.get(CONSTITUTION_SOURCE_DESCRIPTOR_ENV, "").strip()
    if env_path:
        configured.append((CONSTITUTION_SOURCE_DESCRIPTOR_ENV, Path(env_path)))
    if not configured:
        return None

    resolved: set[Path] = set()
    rendered: list[str] = []
    for source, path in configured:
        try:
            resolved_path = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConstitutionSourceError(
                f"Cannot resolve the constitution source descriptor from "
                f"{source} at {path}: {exc}. A configured descriptor that "
                "cannot be read fails closed; it never falls back to the "
                "packaged constitution."
            ) from exc
        resolved.add(resolved_path)
        rendered.append(f"{source}={resolved_path}")
    if len(resolved) != 1:
        raise ConstitutionSourceError(
            "Ambiguous constitution source-descriptor configuration: "
            f"{', '.join(rendered)}. Configure exactly one descriptor file "
            "(or make every source name the same file)."
        )
    return next(iter(resolved))


def pin_source_descriptor_launch_env(
    env: MutableMapping[str, str],
    *,
    explicit_path: str | os.PathLike[str] | None,
) -> Optional[Path]:
    """Carry one agent's descriptor selection into a child process environment.

    A process handoff cannot pass ``constitution_source_descriptor_path`` to
    the child's ``KestrelAgent``, and the child starts with ``multi_agent.toml``
    loading disabled, so the environment variable is the only channel. The
    per-agent setting and the launch environment are resolved here by
    :func:`configured_source_descriptor_path`, under the same conflict rules
    the in-process agent applies, and the child is left exactly one answer:
    the resolved descriptor, or an explicit blank so a ``.env`` loaded later
    without override cannot supply one.

    Raises:
        ConstitutionSourceError: The configuration is ambiguous or names a
            descriptor that does not exist. ``env`` is left unchanged.
    """
    resolved = configured_source_descriptor_path(
        explicit_path=explicit_path, environ=env
    )
    env[CONSTITUTION_SOURCE_DESCRIPTOR_ENV] = (
        "" if resolved is None else str(resolved)
    )
    return resolved


def load_source_descriptor(
    path: Path,
    *,
    trusted_did_document: Mapping[str, Any],
) -> VerifiedSourceDescriptor:
    """Read and verify one descriptor file without touching any agent state."""
    try:
        with Path(path).open("rb") as descriptor_file:
            raw = descriptor_file.read(MAX_SOURCE_DESCRIPTOR_BYTES + 1)
    except OSError as exc:
        raise ConstitutionSourceError(
            f"Cannot read constitution source descriptor {path}: {exc}"
        ) from exc
    if len(raw) > MAX_SOURCE_DESCRIPTOR_BYTES:
        raise ConstitutionSourceError(
            f"Constitution source descriptor {path} exceeds the "
            f"{MAX_SOURCE_DESCRIPTOR_BYTES}-byte maximum."
        )
    try:
        descriptor = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConstitutionSourceError(
            f"Constitution source descriptor {path} is not valid JSON: {exc}"
        ) from exc
    try:
        signer, verification = verify_source_descriptor(
            descriptor, trusted_did_document=trusted_did_document
        )
    except ConstitutionSourceError as exc:
        raise ConstitutionSourceError(
            f"Constitution source descriptor {path} is not trusted: {exc}"
        ) from exc
    return VerifiedSourceDescriptor(
        source_kind=descriptor["source_kind"],
        source_path=descriptor.get("source_path"),
        content_sha256=descriptor["content_sha256"],
        signer=signer,
        verification=verification,
        descriptor_path=str(path),
        descriptor_sha256=hashlib.sha256(raw).hexdigest(),
    )
