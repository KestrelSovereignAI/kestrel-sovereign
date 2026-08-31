"""Verification-document lookup for signed A2A envelopes.

This module answers one question only: which DID document should the envelope
verifier use for a claimed sender DID? Local hybrid identities are resolved
from the live :class:`AgentManager`; optional ``did:web`` lookup is a
per-recipient policy.

Peer authorization is deliberately separate. A valid signature proves control
of a DID, not permission to send work to this recipient. Hosted authorization
is enforced after cryptographic verification by
``a2a.inbound_authorization.RecipientA2ASenderAuthorizer``.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

A2A_PEER_IDENTITY_ROOTS_ENV = "KESTREL_A2A_PEER_IDENTITY_ROOTS"
_MAX_LOCAL_DID_DOCUMENT_BYTES = 1024 * 1024


class ProcessA2ADidResolverConfigurationError(RuntimeError):
    """A process-managed peer verification registry is malformed."""


def _read_local_identity_json(path: Path) -> Optional[Mapping[str, Any]]:
    """Read one bounded, non-symlink public identity document."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not open local A2A DID document %s: %s", path, exc)
        return None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return None
        if os.name == "posix" and (
            opened.st_uid != os.geteuid() or opened.st_mode & 0o022
        ):
            logger.warning(
                "Local A2A DID document is not owner-controlled: %s",
                path,
            )
            return None
        if opened.st_size > _MAX_LOCAL_DID_DOCUMENT_BYTES:
            logger.warning("Local A2A DID document is too large: %s", path)
            return None
        chunks: list[bytes] = []
        remaining = _MAX_LOCAL_DID_DOCUMENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        material = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(material) > _MAX_LOCAL_DID_DOCUMENT_BYTES:
        logger.warning("Local A2A DID document is too large: %s", path)
        return None
    try:
        decoded = json.loads(material.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Local A2A DID document is malformed (%s): %s", path, exc)
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _verification_document_from_material(
    path: Path,
    did: str,
    *,
    material: Optional[Mapping[str, Any]] = None,
) -> Optional[Mapping[str, Any]]:
    if material is None:
        material = _read_local_identity_json(path)
    if material is None:
        return None
    if path.parent.name == "successions":
        document_id = material.get("successor_did")
        verification_methods = material.get("successor_verification_methods")
    else:
        document_id = material.get("id")
        verification_methods = material.get("verificationMethod")
    if document_id != did or not isinstance(verification_methods, list):
        return None
    if not verification_methods or any(
        not isinstance(method, Mapping)
        or not isinstance(method.get("id"), str)
        or not isinstance(method.get("publicKeyMultibase"), str)
        or method.get("controller") != did
        for method in verification_methods
    ):
        logger.warning(
            "Local A2A identity material for %s has invalid verification methods: %s",
            did,
            path,
        )
        return None
    return {
        "id": did,
        "verificationMethod": [dict(method) for method in verification_methods],
    }


class ProcessA2ADidResolver:
    """Resolve subprocess peers from an immutable startup registry.

    The launcher selects the identity roots, but managed peers commonly share
    an OS account and can therefore rewrite one another's public export files.
    Reading those files for every envelope would let a peer replace another
    principal's verification key after startup.  Snapshot the validated public
    documents while the resolver is installed and never consult the writable
    roots on an authorization path again.
    """

    def __init__(
        self,
        identity_roots: tuple[Path, ...],
        *,
        federated_fallback: bool = False,
    ) -> None:
        self._federated_fallback = federated_fallback
        documents: dict[str, list[Mapping[str, Any]]] = {}
        for root in identity_roots:
            candidates = tuple(sorted(root.glob("*_did.json")))
            successions = root / "successions"
            if successions.is_dir() and not successions.is_symlink():
                candidates += tuple(sorted(successions.glob("*.json")))
            for path in candidates:
                material = _read_local_identity_json(path)
                if material is None:
                    continue
                document_id = (
                    material.get("successor_did")
                    if path.parent.name == "successions"
                    else material.get("id")
                )
                if not isinstance(document_id, str) or not document_id:
                    continue
                document = _verification_document_from_material(
                    path,
                    document_id,
                    material=material,
                )
                if document is not None:
                    documents.setdefault(document_id, []).append(
                        deepcopy(document)
                    )
        self._documents = {
            did: tuple(claims) for did, claims in documents.items()
        }

    def resolve(self, did: str) -> Optional[Mapping[str, Any]]:
        if not isinstance(did, str) or not did:
            return None
        documents = self._documents.get(did, ())
        if len(documents) == 1:
            return deepcopy(documents[0])
        if len(documents) > 1:
            logger.warning(
                "A2A DID resolution refused for %s: multiple process identity "
                "roots claim the signing DID",
                did,
            )
            return None
        if self._federated_fallback:
            return HostA2ADidResolver._resolve_federated(did)
        return None


def install_process_a2a_did_resolver(
    agent: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> Optional[ProcessA2ADidResolver]:
    """Install the process-launcher's local peer DID registry when present."""

    environ = os.environ if environment is None else environment
    encoded_roots = environ.get(A2A_PEER_IDENTITY_ROOTS_ENV)
    if encoded_roots is None:
        return None
    if len(encoded_roots.encode("utf-8")) > 64 * 1024:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity-root registry is too large"
        )
    try:
        configured_roots = json.loads(encoded_roots)
    except json.JSONDecodeError as exc:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity-root registry is not valid JSON"
        ) from exc
    if not isinstance(configured_roots, list) or not configured_roots:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity-root registry must be a non-empty list"
        )
    roots: list[Path] = []
    for configured in configured_roots:
        if not isinstance(configured, str) or not configured:
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity roots must be non-empty absolute paths"
            )
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity roots must be non-empty absolute paths"
            )
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)
    if not any(root.is_dir() for root in roots):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity-root registry has no readable directory"
        )
    federated = environ.get("KESTREL_A2A_FEDERATED_DID", "").lower() in {
        "1",
        "true",
        "yes",
    }
    resolver = ProcessA2ADidResolver(
        tuple(roots),
        federated_fallback=federated,
    )
    agent.a2a_did_resolver = resolver.resolve
    logger.info(
        "Process A2A verification DID resolver installed with %d local root(s)",
        len(roots),
    )
    return resolver


def local_a2a_verification_document(
    agent: Any,
    did: str,
) -> Optional[Mapping[str, Any]]:
    """Build the minimal verification document for one loaded identity."""
    identity = getattr(agent, "identity", None)
    if identity is None or not getattr(identity, "is_hybrid", False):
        return None
    if getattr(identity, "signing_did", None) != did:
        return None
    verification_methods = getattr(
        identity, "new_verification_methods", None
    )
    if not verification_methods:
        return None
    return {
        "id": did,
        "verificationMethod": list(verification_methods),
    }


class HostA2ADidResolver:
    """Resolve DID documents for one recipient's envelope verifier."""

    def __init__(
        self,
        manager: Any,
        *,
        recipient: Any = None,
        federated_fallback: bool = False,
    ):
        self._manager = manager
        self._recipient = recipient
        self._federated_fallback = federated_fallback

    def resolve(self, did: str) -> Optional[Mapping[str, Any]]:
        """Return one unambiguous local document, then optional ``did:web``."""
        if not isinstance(did, str) or not did:
            return None

        local_documents = [
            document
            for agent in self._agents()
            if (document := self._document_for_agent(agent, did)) is not None
        ]
        if len(local_documents) == 1:
            return local_documents[0]
        if len(local_documents) > 1:
            logger.warning(
                "A2A DID resolution refused for %s: multiple loaded agents "
                "claim the signing DID",
                did,
            )
            return None

        if self._allows_federated_fallback():
            return self._resolve_federated(did)
        return None

    @staticmethod
    def _document_for_agent(
        agent: Any,
        did: str,
    ) -> Optional[Mapping[str, Any]]:
        return local_a2a_verification_document(agent, did)

    def _agents(self) -> tuple[Any, ...]:
        agents = self._manager.list_agents()
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        return tuple(agent for agent in iterable if agent is not None)

    def _allows_federated_fallback(self) -> bool:
        """Read federation policy from this resolver's recipient."""
        if self._recipient is None:
            return self._federated_fallback
        configured = getattr(
            self._recipient, "a2a_federated_did_fallback", None
        )
        if configured is None:
            return self._federated_fallback
        return configured is True

    @staticmethod
    def _resolve_federated(did: str) -> Optional[Mapping[str, Any]]:
        if not did.startswith("did:web:"):
            return None
        try:
            from kestrel_sovereign.identity.did_web import (
                resolve as did_web_resolve,
            )

            return did_web_resolve(did)
        except Exception as exc:  # noqa: BLE001 - resolution is a trust boundary
            logger.warning(
                "A2A federated did:web resolution failed for %s: %s",
                did,
                exc,
            )
            return None


def install_a2a_did_resolver(
    manager: Any,
    *,
    recipient: Any = None,
    federated_fallback: bool = False,
) -> HostA2ADidResolver:
    """Install a distinct verification resolver on each selected recipient.

    The host registration hook passes only the newly registered ``recipient``.
    Omitting it retains explicit batch installation for standalone embeddings.
    """
    if recipient is not None:
        recipients = (recipient,)
    else:
        agents = manager.list_agents()
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        recipients = tuple(agent for agent in iterable if agent is not None)

    first_resolver: Optional[HostA2ADidResolver] = None
    for current_recipient in recipients:
        resolver = HostA2ADidResolver(
            manager,
            recipient=current_recipient,
            federated_fallback=federated_fallback,
        )
        current_recipient.a2a_did_resolver = resolver.resolve
        if first_resolver is None:
            first_resolver = resolver

    if first_resolver is None:
        first_resolver = HostA2ADidResolver(
            manager,
            federated_fallback=federated_fallback,
        )

    logger.info(
        "A2A verification DID resolver installed on %d recipient(s) "
        "(federated_fallback=%s)",
        len(recipients),
        federated_fallback,
    )
    return first_resolver


__all__ = [
    "A2A_PEER_IDENTITY_ROOTS_ENV",
    "HostA2ADidResolver",
    "ProcessA2ADidResolver",
    "ProcessA2ADidResolverConfigurationError",
    "install_a2a_did_resolver",
    "install_process_a2a_did_resolver",
    "local_a2a_verification_document",
]
