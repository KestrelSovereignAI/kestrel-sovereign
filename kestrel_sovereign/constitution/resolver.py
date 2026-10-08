"""Single production resolver for the governing constitution bytes.

Inception, explicit verification (``!verify-constitution``), the periodic
integrity audit, and reanchor MUST resolve the authoritative governing
constitution through this module so they can never diverge (issue #2463).

Two questions are answered here, in order:

1. **Which source governs?** :func:`resolve_governing_source`. By default the
   packaged canonical constitution at ``config.CONSTITUTION_PATH``
   (``kestrel_sovereign/data/KESTREL_CONSTITUTION.md``) — **not** the
   documentation copy under ``docs/principles/``, which carries OKF YAML
   frontmatter and is free to drift. An operator may instead configure a
   Sovereign-signed source descriptor (#2553, see
   :mod:`kestrel_sovereign.constitution.source_descriptor`), which selects the
   package or an external file and pins its content digest. Selection reads
   operator configuration and the out-of-DB trust root only; nothing in the
   agent's graph database is consulted.
2. **What are its bytes?** :func:`resolve_governing_constitution_bytes`. When
   an agent has an active Amendment VIII emancipation contract, the governing
   bytes are the **rendered active form** — matching exactly what inception
   anchored — rather than the dormant canonical text.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from .emancipation import EmancipationContract, apply_emancipation
from .source_descriptor import (
    SOURCE_KIND_PACKAGE,
    ConstitutionSourceError,
    VerifiedSourceDescriptor,
    configured_source_descriptor_path,
    load_source_descriptor,
)


def governing_constitution_path() -> str:
    """Return the packaged governing constitution path.

    Deferred import of ``config`` keeps this module importable without
    triggering path resolution at import time.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH

    return CONSTITUTION_PATH


@dataclass(frozen=True)
class GoverningSource:
    """The source whose bytes govern an agent, and how it was chosen.

    ``content_sha256`` is the raw-source digest a verified descriptor pinned;
    None means the unpinned packaged default (no descriptor configured).
    """

    kind: str
    path: str
    content_sha256: Optional[str] = None
    descriptor: Optional[VerifiedSourceDescriptor] = None

    def receipt_fields(self) -> dict[str, Any]:
        """Audit fields for a receipt recording an anchoring under this source.

        Evidence only: no code path reads these back to choose a source.
        """
        fields: dict[str, Any] = {"source_kind": self.kind}
        if self.descriptor is not None:
            fields.update(
                {
                    "source_content_sha256": self.descriptor.content_sha256,
                    "source_descriptor_path": self.descriptor.descriptor_path,
                    "source_descriptor_sha256": self.descriptor.descriptor_sha256,
                    "source_descriptor_signer": self.descriptor.signer,
                }
            )
        return fields


def resolve_governing_source(
    *,
    descriptor_path: str | os.PathLike[str] | None = None,
    trust_root_path: str | os.PathLike[str] | None = None,
    agent_dids: set[str] | frozenset[str] = frozenset(),
    environ: Mapping[str, str] | None = None,
) -> GoverningSource:
    """Decide which source governs, from out-of-DB configuration only.

    With no descriptor configured (neither ``descriptor_path`` nor
    ``KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH``), the packaged source
    governs and no trust root is needed — unchanged behavior for every agent
    that never opted in. With one configured, it must verify against the
    operator-pinned Sovereign trust root, the same
    :func:`~kestrel_sovereign.constitution.trust_root.load_sovereign_trust_root`
    that authorizes reanchor (#2499). ``agent_dids`` lets that resolver refuse
    an agent-owned DID as the root.

    Raises:
        ConstitutionSourceError: The descriptor configuration is ambiguous,
            missing, unreadable, malformed, unsigned, or signed by anyone but
            the trust root. Never answered by falling back to the package.
        SovereignTrustRootError: A descriptor is configured but no usable
            trust root is. Both are ``ValueError`` subclasses.
    """
    configured = configured_source_descriptor_path(
        explicit_path=descriptor_path, environ=environ
    )
    if configured is None:
        return GoverningSource(
            kind=SOURCE_KIND_PACKAGE, path=governing_constitution_path()
        )

    from .trust_root import load_sovereign_trust_root

    trusted_did_document = load_sovereign_trust_root(
        explicit_path=trust_root_path,
        environ=environ,
        agent_dids=agent_dids,
    )
    verified = load_source_descriptor(
        configured, trusted_did_document=trusted_did_document
    )
    if verified.source_kind == SOURCE_KIND_PACKAGE:
        path = governing_constitution_path()
    else:
        # Validated absolute by ``load_source_descriptor``.
        path = str(verified.source_path)
    return GoverningSource(
        kind=verified.source_kind,
        path=path,
        content_sha256=verified.content_sha256,
        descriptor=verified,
    )


def is_authoritative_governing_source(
    constitution_path: Optional[str],
    source: Optional[GoverningSource] = None,
) -> bool:
    """Return True when ``constitution_path`` is the authoritative governing source.

    The periodic integrity audit ALWAYS recomputes the governing hash from the
    resolved governing source. Any inception / offline-reanchor that anchors
    bytes from a *different* path manufactures an agent guaranteed to fail its
    next audit and Safe-Mode, so the production paths must refuse
    non-authoritative inputs (issue #2463 review).

    ``source`` is the :class:`GoverningSource` the caller resolved; without
    one, the packaged default is authoritative. ``None`` for
    ``constitution_path`` means "use the governing source" and is therefore
    authoritative. Otherwise the path is compared on ``os.path.realpath`` so
    symlinks, ``..`` segments, and differently-spelled-but-equivalent paths
    still count. A custom governing source is expressed by a Sovereign-signed
    source descriptor (#2553), never by an unsigned path override; tests may
    still monkeypatch ``config.CONSTITUTION_PATH``, which this reads through
    ``governing_constitution_path()``.
    """
    if constitution_path is None:
        return True
    authoritative = (
        source.path if source is not None else governing_constitution_path()
    )
    return os.path.realpath(constitution_path) == os.path.realpath(authoritative)


def resolve_governing_constitution_bytes(
    contract: Optional[EmancipationContract] = None,
    *,
    constitution_path: Optional[str] = None,
    source: Optional[GoverningSource] = None,
    content: Optional[bytes] = None,
) -> bytes:
    """Return the authoritative governing constitution bytes.

    Reads the governing source and, when ``contract`` is an active Amendment
    VIII emancipation contract, renders its active form so the result matches
    what inception anchored for that agent.

    Args:
        contract: The agent's anchored emancipation contract, or None. A
            dormant / None contract yields the canonical text unchanged.
        constitution_path: Explicit file to read, bypassing source selection.
            For tests and single-file diagnostics; production callers pass
            ``source``.
        source: The :class:`GoverningSource` from
            :func:`resolve_governing_source`. When it carries a pinned
            ``content_sha256``, the raw bytes must hash to it.
        content: The source's raw bytes as they will be when the agent next
            starts, instead of reading the path now. A deploy gate passes the
            packaged constitution of a revision that is not checked out yet
            (#3517), so the bytes meet exactly the tests and rendering the
            startup audit applies to the file on disk.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        OSError: If the resolved path exists but cannot be read (e.g. a
            permission denial).
        ValueError: If the resolved source is empty / whitespace-only — an
            authoritative governing source can never be blank, so an empty
            read is treated as an unreadable/ambiguous source rather than a
            valid (hashable) constitution. Also
            :class:`~.source_descriptor.ConstitutionSourceError` when the
            bytes no longer match a descriptor's pinned digest, and
            :class:`~.emancipation.AmbiguousAmendmentVIII` when an active
            contract must be substituted into a text carrying more than one
            Amendment VIII heading: which section is the amendment has no
            answer, which is ambiguity in exactly the sense this contract
            already fails closed on.

    Callers rely on these raising so they FAIL CLOSED: a governing source that
    is missing, unreadable, drifted, or ambiguous must never be silently
    substituted or treated as "verified" (issue #2463, #2553). The periodic
    integrity audit converts any of these into an integrity failure → Safe
    Mode. A descriptor-selected external source gets exactly the packaged
    source's semantics.
    """
    if source is not None and constitution_path is not None:
        raise ValueError(
            "Pass either a resolved governing source or an explicit "
            "constitution_path, not both."
        )
    if source is not None:
        path = source.path
    else:
        path = constitution_path or governing_constitution_path()
    if content is None:
        # ``open`` raises FileNotFoundError (missing) or OSError/PermissionError
        # (unreadable) — both propagate so callers fail closed.
        with open(path, "rb") as f:
            content = f.read()
    if not content.strip():
        # A blank authoritative source is not a valid constitution; refuse to
        # hand back empty bytes that would hash to a spurious "valid" digest.
        raise ValueError(
            f"Governing constitution at {path} is empty or unreadable; "
            f"refusing to treat a blank source as authoritative."
        )
    if source is not None and source.content_sha256 is not None:
        actual = hashlib.sha256(content).hexdigest()
        if actual != source.content_sha256:
            raise ConstitutionSourceError(
                f"Governing constitution at {path} hashes to {actual[:16]}…, "
                f"but its Sovereign-signed source descriptor pins "
                f"{source.content_sha256[:16]}…: the {source.kind} source "
                f"changed after the descriptor was signed."
            )
    if contract is not None and contract.enabled:
        content = apply_emancipation(content.decode("utf-8"), contract).encode("utf-8")
    return content
