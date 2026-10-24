"""Complete-request hybrid proofs for the A2A signed-sender-v1 extension.

This extension is separate from standard Agent Card JWS. Its HTTP header is
three unpadded base64url segments: JCS claims, classical signature, PQ
signature. Every claim, including both signer identifiers, is signed. The
request digest covers the complete official ProtoJSON request, after tenant
binding and before the SDK generates task/context IDs. Verification establishes
identity only; recipient authorization and live identity-epoch checks follow.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import rfc8785
from google.protobuf.message import Message as ProtoMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kestrel_sovereign.a2a.extensions import SIGNED_SENDER_URI
from kestrel_sovereign.a2a.native_input import (
    NativeSenderProofError,
    native_request_digest,
)
from kestrel_sovereign.identity.hybrid_keypair import (
    HybridKeypair,
    sign_hybrid,
    verify_hybrid,
)
from kestrel_sovereign.security.crypto_suite import ALG_ED25519, ALG_ML_DSA_65
from kestrel_sovereign.security.verify_policy import VerifyPolicy

Resolver = Callable[
    [str], Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]
]


class ReplayNonceStore(Protocol):
    """Shared replay-nonce reservation backend."""

    async def reserve(
        self,
        sender: str,
        nonce: str,
        *,
        now_ts: float,
        ttl_seconds: int,
    ) -> bool:
        """Reserve ``(sender, nonce)``. Return False when already consumed."""
        ...


def verification_document_fingerprint(
    did_document: Mapping[str, Any],
) -> str:
    """Digest exactly the DID-document fields used by native proof verification."""
    relevant = {
        "id": did_document.get("id"),
        "verificationMethod": did_document.get("verificationMethod") or [],
    }
    canonical = json.dumps(
        relevant,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


SENDER_PROOF_HEADER = "Kestrel-A2A-Sender-Proof"
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+\Z")


class NativeProofPolicy(BaseModel):
    """Host-configured verification limits, shared by signing and verification."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_age_seconds: int = Field(default=300, gt=0)
    max_header_bytes: int = Field(default=8192, gt=0)
    verify_policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED


class _Signer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    alg: str = Field(min_length=1, max_length=64)
    kid: str = Field(min_length=1, max_length=256)


class _Claims(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    protocol: Literal["1.0"]
    extension: str
    sender: str = Field(min_length=1, max_length=2048)
    recipient: str = Field(min_length=1, max_length=2048)
    operation: str = Field(min_length=1, max_length=128)
    request_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    timestamp: int = Field(ge=0, le=2**53 - 1)
    nonce: str = Field(pattern=r"^[a-f0-9]{32}$")
    extensions: list[str]
    signers: list[_Signer] = Field(min_length=2, max_length=2)


@dataclass(frozen=True)
class NativeVerifiedSender:
    """Cryptographic evidence; this object confers no recipient permissions."""

    sender: str
    nonce: str
    verification_document_fingerprint: str


def _extension_set(extensions: Iterable[str]) -> list[str]:
    values = list(extensions)
    if any(not isinstance(value, str) or not value for value in values):
        raise NativeSenderProofError("Invalid activated A2A extension identifier")
    result = sorted(set(values))
    if SIGNED_SENDER_URI not in result:
        raise NativeSenderProofError("Signed sender extension was not activated")
    return result


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not _BASE64URL.fullmatch(value):
        raise NativeSenderProofError("Malformed sender proof encoding")
    try:
        result = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as error:
        raise NativeSenderProofError("Malformed sender proof encoding") from error
    if _encode(result) != value:
        raise NativeSenderProofError("Noncanonical sender proof encoding")
    return result


def _check_header_size(header: str, policy: NativeProofPolicy) -> None:
    if (
        not isinstance(header, str)
        or not header.isascii()
        or len(header) > policy.max_header_bytes
    ):
        raise NativeSenderProofError(
            "Sender proof exceeds the configured ASCII header limit"
        )


def sign_native_request(
    keypair: HybridKeypair,
    params: ProtoMessage,
    *,
    sender: str,
    recipient: str,
    operation: str,
    timestamp: int,
    nonce: str,
    extensions: Iterable[str],
    classical_kid: str,
    pq_kid: str,
    policy: NativeProofPolicy,
) -> str:
    """Produce one compact extension header without changing the native input."""
    if (keypair.classical.suite_id, keypair.pq.suite_id) != (
        ALG_ED25519,
        ALG_ML_DSA_65,
    ):
        raise NativeSenderProofError("Signed sender v1 requires Ed25519 and ML-DSA-65")
    try:
        claims = _Claims(
            version=1,
            protocol="1.0",
            extension=SIGNED_SENDER_URI,
            sender=sender,
            recipient=recipient,
            operation=operation,
            request_digest=native_request_digest(params, operation),
            timestamp=timestamp,
            nonce=nonce,
            extensions=_extension_set(extensions),
            signers=[
                _Signer(alg=ALG_ED25519, kid=classical_kid),
                _Signer(alg=ALG_ML_DSA_65, kid=pq_kid),
            ],
        )
    except ValidationError as error:
        raise NativeSenderProofError("Invalid native sender proof claims") from error
    encoded = rfc8785.dumps(claims.model_dump(mode="json"))
    signatures = sign_hybrid(
        encoded, keypair, classical_kid=classical_kid, pq_kid=pq_kid
    )
    header = ".".join(
        [
            _encode(encoded),
            *(_encode(bytes.fromhex(item["sig"])) for item in signatures),
        ]
    )
    _check_header_size(header, policy)
    return header


async def verify_native_request(
    headers: list[str],
    params: ProtoMessage,
    *,
    recipient: str,
    operation: str,
    extensions: Iterable[str],
    resolver: Resolver,
    replay_store: ReplayNonceStore,
    policy: NativeProofPolicy,
    now_ts: int,
) -> NativeVerifiedSender:
    """Authenticate, then spend the nonce in the mandatory shared store.

    There is no unsigned or process-local degradation. Storage/resolver control
    errors propagate; an uncertain reservation is never released or retried by
    this function. A lost response does not authorize replaying the operation.
    """
    if len(headers) != 1:
        raise NativeSenderProofError("Exactly one sender proof header is required")
    header = headers[0]
    _check_header_size(header, policy)
    segments = header.split(".")
    if len(segments) != 3:
        raise NativeSenderProofError("Malformed sender proof segments")
    encoded, classical, pq = map(_decode, segments)
    try:
        claims = _Claims.model_validate_json(encoded)
    except ValidationError as error:
        raise NativeSenderProofError("Malformed sender proof claims") from error
    if rfc8785.dumps(claims.model_dump(mode="json")) != encoded:
        raise NativeSenderProofError("Sender proof claims are not canonical")
    if claims.extension != SIGNED_SENDER_URI:
        raise NativeSenderProofError("Sender proof uses another extension domain")
    if claims.recipient != recipient or claims.operation != operation:
        raise NativeSenderProofError(
            "Sender proof targets another recipient or operation"
        )
    if claims.extensions != _extension_set(extensions):
        raise NativeSenderProofError("Sender proof activates different extensions")
    if claims.request_digest != native_request_digest(params, operation):
        raise NativeSenderProofError(
            "Sender proof does not bind the complete native request"
        )
    if abs(now_ts - claims.timestamp) > policy.max_age_seconds:
        raise NativeSenderProofError(
            "Sender proof timestamp is outside the permitted window"
        )
    if [item.alg for item in claims.signers] != [ALG_ED25519, ALG_ML_DSA_65]:
        raise NativeSenderProofError("Unsupported sender proof signature algorithms")
    document = resolver(claims.sender)
    if inspect.isawaitable(document):
        document = await document
    if not isinstance(document, Mapping) or document.get("id") != claims.sender:
        raise NativeSenderProofError("Sender DID document is unavailable or mismatched")
    signatures = [
        {"alg": signer.alg, "kid": signer.kid, "sig": signature.hex()}
        for signer, signature in zip(claims.signers, (classical, pq), strict=True)
    ]
    verdict = verify_hybrid(
        encoded,
        signatures,
        document.get("verificationMethod") or [],
        policy=policy.verify_policy,
    )
    if not verdict.ok:
        raise NativeSenderProofError("Sender proof signature verification failed")
    try:
        fingerprint = verification_document_fingerprint(document)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NativeSenderProofError(
            "Sender DID document cannot be fingerprinted"
        ) from error
    reserved = await replay_store.reserve(
        claims.sender,
        claims.nonce,
        now_ts=now_ts,
        ttl_seconds=2 * policy.max_age_seconds,
    )
    if reserved is not True:
        raise NativeSenderProofError("Sender proof nonce was already consumed")
    return NativeVerifiedSender(claims.sender, claims.nonce, fingerprint)
