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

import hashlib
import hmac
import json
import logging
import os
import stat
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

A2A_PEER_IDENTITY_ROOTS_ENV = "KESTREL_A2A_PEER_IDENTITY_ROOTS"
A2A_PEER_IDENTITY_DOCUMENTS_ENV = "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS"
A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV = (
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS_FILE"
)
A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV = (
    "KESTREL_A2A_PEER_IDENTITY_DOCUMENTS_SHA256"
)
_MAX_PROCESS_REGISTRY_BYTES = 128 * 1024
_MAX_WINDOWS_ENV_VALUE_CHARS = 32_766
_PROCESS_REGISTRY_DIRECTORY = ".kestrel-launch"
_PROCESS_REGISTRY_PREFIX = "a2a-peer-identities-"


class ProcessA2ADidResolverConfigurationError(RuntimeError):
    """A process-managed peer verification registry is malformed."""


def _platform_requires_process_registry_file(encoded_documents: str) -> bool:
    """Whether registry JSON cannot safely travel in one environment value.

    Modern Windows permits large Unicode environment blocks, but a single
    user-defined variable is limited to 32,767 characters including its NUL.
    POSIX launchers keep the established inline contract, whose explicit
    128-KiB cap is below the per-entry limit on supported hosts.
    """

    return (
        sys.platform == "win32"
        and len(encoded_documents) > _MAX_WINDOWS_ENV_VALUE_CHARS
    )


def stage_process_a2a_did_registry(
    documents: list[Mapping[str, Any]],
    *,
    launch_root: Path,
) -> tuple[dict[str, str], Optional[Path]]:
    """Prepare an authenticated, platform-safe child registry handoff.

    The returned path, when present, is a one-shot launcher-created file. Its
    content is public key material, but its authority comes from the SHA-256
    digest inherited in the child's environment: a sibling cannot substitute
    different keys by rewriting the file. The child verifies and removes it
    during resolver installation.
    """

    encoded_text = json.dumps(
        documents,
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = encoded_text.encode("utf-8")
    if len(encoded) > _MAX_PROCESS_REGISTRY_BYTES:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry is too large"
        )
    if not _platform_requires_process_registry_file(encoded_text):
        return {A2A_PEER_IDENTITY_DOCUMENTS_ENV: encoded_text}, None

    registry_dir = Path(launch_root) / _PROCESS_REGISTRY_DIRECTORY
    registry_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        registry_dir.chmod(0o700)
    fd, raw_path = tempfile.mkstemp(
        prefix=_PROCESS_REGISTRY_PREFIX,
        suffix=".json",
        dir=registry_dir,
    )
    registry_path = Path(raw_path)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as registry_file:
            fd = -1
            registry_file.write(encoded)
            registry_file.flush()
            os.fsync(registry_file.fileno())
    except BaseException:
        if fd >= 0:
            os.close(fd)
        registry_path.unlink(missing_ok=True)
        raise

    return (
        {
            A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV: str(registry_path.resolve()),
            A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV: hashlib.sha256(
                encoded
            ).hexdigest(),
        },
        registry_path,
    )


def _read_authenticated_process_registry(
    registry_path_value: str,
    expected_digest: str,
) -> str:
    """Read and consume one launcher-created registry file."""

    registry_path = Path(registry_path_value)
    if not registry_path.is_absolute():
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file path must be absolute"
        )
    if (
        registry_path.parent.name != _PROCESS_REGISTRY_DIRECTORY
        or not registry_path.name.startswith(_PROCESS_REGISTRY_PREFIX)
    ):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file is not launcher-owned"
        )
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file digest is invalid"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(registry_path, flags)
    except (OSError, ValueError) as exc:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file could not be opened"
        ) from exc
    should_unlink = False
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity registry file is not a regular file"
            )
        should_unlink = True
        if os.name == "posix" and (
            file_stat.st_uid != os.geteuid() or file_stat.st_mode & 0o077
        ):
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity registry file permissions are unsafe"
            )
        with os.fdopen(fd, "rb") as registry_file:
            fd = -1
            encoded = registry_file.read(_MAX_PROCESS_REGISTRY_BYTES + 1)
        if len(encoded) > _MAX_PROCESS_REGISTRY_BYTES:
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity registry is too large"
            )
        actual_digest = hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity registry file failed authentication"
            )
        try:
            decoded = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProcessA2ADidResolverConfigurationError(
                "A2A peer identity registry is not valid UTF-8"
            ) from exc
        return decoded
    finally:
        if fd >= 0:
            os.close(fd)
        if should_unlink:
            registry_path.unlink(missing_ok=True)


def _validated_verification_document(
    material: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Normalize one launcher-attested hybrid verification document."""

    did = material.get("id")
    verification_methods = material.get("verificationMethod")
    if (
        not isinstance(did, str)
        or not did
        or not isinstance(verification_methods, list)
    ):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry contains an invalid document"
        )
    if not verification_methods or any(
        not isinstance(method, Mapping)
        or not isinstance(method.get("id"), str)
        or not isinstance(method.get("publicKeyMultibase"), str)
        or method.get("controller") != did
        for method in verification_methods
    ):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry contains invalid verification methods"
        )
    return {
        "id": did,
        "verificationMethod": [dict(method) for method in verification_methods],
    }


def launcher_attested_a2a_verification_document(
    identity_root: Path,
    *,
    expected_agent_did: str,
    master_key: str | bytes | None,
) -> Optional[Mapping[str, Any]]:
    """Load one peer's public keys through its custody and anchor checks.

    A sibling-writable DID file is not authority. The launcher first loads the
    complete runtime identity, which proves that its encrypted private keys
    match the published verification methods, then binds that identity to the
    durable DID read independently from the peer's local anchor.
    """

    root = Path(identity_root).expanduser().resolve()
    legacy_documents = sorted(root.glob("kestrel_0x*.json"))
    born_hybrid_documents = sorted(root.glob("*_did.json"))
    if not legacy_documents and not born_hybrid_documents:
        return None
    if (
        len(legacy_documents) > 1
        or len(born_hybrid_documents) > 1
        or (legacy_documents and born_hybrid_documents)
    ):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity custody contains ambiguous DID documents"
        )
    if not isinstance(expected_agent_did, str) or not expected_agent_did:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity custody has no durable anchor"
        )

    from kestrel_sovereign.identity.runtime_identity import load_agent_identity

    legacy_key_id = legacy_documents[0].stem if legacy_documents else None
    try:
        identity = load_agent_identity(
            legacy_key_id,
            storage_dir=root,
            master_key=master_key,
        )
    except Exception as exc:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity custody failed cryptographic validation "
            f"({type(exc).__name__})"
        ) from exc
    bound_dids = {
        candidate
        for candidate in (identity.legacy_did, identity.new_did)
        if isinstance(candidate, str) and candidate
    }
    if expected_agent_did not in bound_dids:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity custody does not match its durable anchor"
        )
    if not identity.is_hybrid:
        # Legacy peers do not emit signed A2A principal-action envelopes.
        return None
    return _validated_verification_document(
        {
            "id": identity.signing_did,
            "verificationMethod": identity.new_verification_methods,
        }
    )


class ProcessA2ADidResolver:
    """Resolve subprocess peers from launcher-attested immutable documents."""

    def __init__(
        self,
        documents: tuple[Mapping[str, Any], ...],
        *,
        federated_fallback: bool = False,
    ) -> None:
        self._federated_fallback = federated_fallback
        verified: dict[str, Mapping[str, Any]] = {}
        for material in documents:
            document = _validated_verification_document(material)
            did = str(document["id"])
            if did in verified:
                raise ProcessA2ADidResolverConfigurationError(
                    "A2A peer identity registry contains a duplicate signing DID"
                )
            verified[did] = deepcopy(document)
        self._documents = verified

    def resolve(self, did: str) -> Optional[Mapping[str, Any]]:
        if not isinstance(did, str) or not did:
            return None
        document = self._documents.get(did)
        if document is not None:
            return deepcopy(document)
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
    if environ.get(A2A_PEER_IDENTITY_ROOTS_ENV) is not None:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity roots are not an attested registry"
        )
    encoded_documents = environ.get(A2A_PEER_IDENTITY_DOCUMENTS_ENV)
    registry_file = environ.get(A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV)
    registry_digest = environ.get(A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV)
    if encoded_documents is not None and registry_file is not None:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry has ambiguous launch sources"
        )
    if registry_file is None and registry_digest is not None:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file is missing"
        )
    if registry_file is not None and registry_digest is None:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry file digest is missing"
        )
    if encoded_documents is None and registry_file is None:
        return None
    if registry_file is not None:
        assert registry_digest is not None
        encoded_documents = _read_authenticated_process_registry(
            registry_file,
            registry_digest,
        )
    assert encoded_documents is not None
    if len(encoded_documents.encode("utf-8")) > _MAX_PROCESS_REGISTRY_BYTES:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry is too large"
        )
    try:
        configured_documents = json.loads(encoded_documents)
    except json.JSONDecodeError as exc:
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry is not valid JSON"
        ) from exc
    if not isinstance(configured_documents, list):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry must be a list"
        )
    if any(not isinstance(document, Mapping) for document in configured_documents):
        raise ProcessA2ADidResolverConfigurationError(
            "A2A peer identity registry entries must be documents"
        )
    federated = environ.get("KESTREL_A2A_FEDERATED_DID", "").lower() in {
        "1",
        "true",
        "yes",
    }
    resolver = ProcessA2ADidResolver(
        tuple(configured_documents),
        federated_fallback=federated,
    )
    agent.a2a_did_resolver = resolver.resolve
    if registry_file is not None and environ is os.environ:
        # Feature initialization is intentionally repeatable. Once the
        # one-shot file has been authenticated and consumed, remove its stale
        # coordinates so a later PeersFeature.initialize() keeps the resolver
        # it already installed instead of trying to reopen a deleted path.
        os.environ.pop(A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV, None)
        os.environ.pop(A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV, None)
    logger.info(
        "Process A2A verification DID resolver installed with %d "
        "launcher-attested document(s)",
        len(configured_documents),
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
    "A2A_PEER_IDENTITY_DOCUMENTS_ENV",
    "A2A_PEER_IDENTITY_DOCUMENTS_FILE_ENV",
    "A2A_PEER_IDENTITY_DOCUMENTS_SHA256_ENV",
    "A2A_PEER_IDENTITY_ROOTS_ENV",
    "HostA2ADidResolver",
    "ProcessA2ADidResolver",
    "ProcessA2ADidResolverConfigurationError",
    "install_a2a_did_resolver",
    "install_process_a2a_did_resolver",
    "launcher_attested_a2a_verification_document",
    "local_a2a_verification_document",
    "stage_process_a2a_did_registry",
]
