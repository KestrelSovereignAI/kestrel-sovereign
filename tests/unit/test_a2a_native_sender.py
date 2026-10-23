"""Native complete-request proof, real hybrid keys and durable replay tests."""

import asyncio
import base64
import hashlib
import json

import pytest
import rfc8785
from kestrel_sdk.native_a2a.types import GetTaskRequest, SendMessageRequest
from google.protobuf.json_format import MessageToDict, ParseDict

from kestrel_sovereign.a2a.native_sender import (
    SIGNED_SENDER_URI,
    NativeProofPolicy,
    NativeSenderProofError,
    native_request_digest,
    sign_native_request,
    verify_native_request,
)
from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.identity.hybrid_keypair import (
    generate_hybrid_keypair,
    sign_hybrid,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.utils.native_a2a import signing_identity

SENDER = "did:web:sender.example:agents:alpha"
RECIPIENT = "did:web:recipient.example:agents:beta"
NOW = 1_700_000_000
NONCE = "a" * 32
EXTENSIONS = [SIGNED_SENDER_URI, "https://example.test/causation-v1"]
POLICY = NativeProofPolicy()


@pytest.fixture(scope="module")
def identity():
    return signing_identity(SENDER)


@pytest.fixture
def request_message():
    return ParseDict(
        {
            "tenant": "tenant-alpha",
            "message": {
                "messageId": "message-1",
                "contextId": "context-1",
                "role": "ROLE_USER",
                "parts": [
                    {"text": "hello"},
                    {
                        "raw": "eA==",
                        "filename": "file.bin",
                        "mediaType": "application/octet-stream",
                    },
                    {"data": {"value": 0.000001}},
                ],
                "metadata": {"question": {"id": "question-1", "replyExpected": True}},
            },
            "configuration": {"returnImmediately": True, "historyLength": 0},
            "metadata": {"causation": [{"agent": "parent"}]},
        },
        SendMessageRequest(),
    )


def _sign(identity, params, **changes):
    options = {
        "sender": SENDER,
        "recipient": RECIPIENT,
        "operation": "SendMessage",
        "timestamp": NOW,
        "nonce": NONCE,
        "extensions": EXTENSIONS,
        "classical_kid": "key-1",
        "pq_kid": "key-2",
        "policy": POLICY,
    }
    options.update(changes)
    return sign_native_request(identity[0], params, **options)


async def _verify(identity, params, headers, store, **changes):
    options = {
        "recipient": RECIPIENT,
        "operation": "SendMessage",
        "extensions": EXTENSIONS,
        "resolver": lambda did: identity[1],
        "replay_store": store,
        "policy": POLICY,
        "now_ts": NOW,
    }
    options.update(changes)
    return await verify_native_request(headers, params, **options)


class _ForbiddenStore:
    async def reserve(self, *args, **kwargs):
        raise AssertionError("Invalid proof reached replay reservation")


def _decode(segment):
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _resign(identity, encoded):
    signatures = sign_hybrid(encoded, identity[0])
    return ".".join(
        [
            _encode(encoded),
            *(_encode(bytes.fromhex(item["sig"])) for item in signatures),
        ]
    )


def test_full_request_digest_uses_actual_protojson_presence(request_message):
    actual = MessageToDict(request_message)
    assert actual["configuration"]["historyLength"] == 0
    assert "returnImmediately" in actual["configuration"]
    assert (
        native_request_digest(request_message, "SendMessage")
        == hashlib.sha256(rfc8785.dumps(actual)).hexdigest()
    )
    request_message.configuration.ClearField("history_length")
    assert "historyLength" not in MessageToDict(request_message)["configuration"]


def test_canonical_request_vector_retains_explicit_optional_zero():
    params = GetTaskRequest(id="task", history_length=0)
    expected = b'{"historyLength":0,"id":"task"}'
    assert (
        native_request_digest(params, "GetTask") == hashlib.sha256(expected).hexdigest()
    )
    params.ClearField("history_length")
    assert (
        native_request_digest(params, "GetTask")
        == hashlib.sha256(b'{"id":"task"}').hexdigest()
    )


def test_compact_claims_have_independent_ed25519_signature(identity, request_message):
    from cryptography.exceptions import InvalidSignature

    header = _sign(identity, request_message)
    segments = header.split(".")
    assert len(segments) == 3 and len(header) < POLICY.max_header_bytes
    claims = _decode(segments[0])
    expected = {
        "version": 1,
        "protocol": "1.0",
        "extension": SIGNED_SENDER_URI,
        "sender": SENDER,
        "recipient": RECIPIENT,
        "operation": "SendMessage",
        "request_digest": native_request_digest(request_message, "SendMessage"),
        "timestamp": NOW,
        "nonce": NONCE,
        "extensions": sorted(EXTENSIONS),
        "signers": [
            {"alg": "ed25519", "kid": "key-1"},
            {"alg": "ml-dsa-65", "kid": "key-2"},
        ],
    }
    assert claims == rfc8785.dumps(expected)
    # Verify directly with cryptography, independently of the production
    # shared hybrid verifier. Claim changes must invalidate that signature.
    identity[0].classical.public_key.verify(_decode(segments[1]), claims)
    with pytest.raises(InvalidSignature):
        identity[0].classical.public_key.verify(_decode(segments[1]), claims + b" ")


@pytest.mark.asyncio
async def test_shared_store_spends_nonce_across_independent_verifiers(
    identity, request_message, tmp_path
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "native-replay.db"))
    try:
        header = _sign(identity, request_message)
        before = request_message.SerializeToString()
        verified = await _verify(
            identity, request_message, [header], SharedReplayNonceStore(db)
        )
        assert verified.sender == SENDER and verified.nonce == NONCE
        assert len(verified.verification_document_fingerprint) == 64
        assert request_message.SerializeToString() == before
        with pytest.raises(NativeSenderProofError, match="consumed"):
            await _verify(
                identity, request_message, [header], SharedReplayNonceStore(db)
            )
    finally:
        await db.close()


@pytest.mark.parametrize(
    "target",
    [
        "tenant",
        "text",
        "raw",
        "filename",
        "media_type",
        "data",
        "question",
        "causation",
        "configuration",
        "presence",
    ],
)
@pytest.mark.asyncio
async def test_every_native_payload_dimension_is_bound(
    identity, request_message, target
):
    header = _sign(identity, request_message)
    if target == "tenant":
        request_message.tenant = "another-tenant"
    elif target == "text":
        request_message.message.parts[0].text = "changed"
    elif target == "raw":
        request_message.message.parts[1].raw = b"changed"
    elif target == "filename":
        request_message.message.parts[1].filename = "changed.bin"
    elif target == "media_type":
        request_message.message.parts[1].media_type = "text/plain"
    elif target == "data":
        ParseDict({"value": 2}, request_message.message.parts[2].data)
    elif target == "question":
        request_message.message.metadata.update({"question": {"id": "other"}})
    elif target == "causation":
        request_message.metadata.update({"causation": []})
    elif target == "configuration":
        request_message.configuration.return_immediately = False
    elif target == "presence":
        request_message.configuration.ClearField("history_length")
    with pytest.raises(NativeSenderProofError, match="complete native request"):
        await _verify(identity, request_message, [header], _ForbiddenStore())


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ({"recipient": "did:web:other.example"}, "recipient"),
        ({"operation": "SendStreamingMessage"}, "operation"),
        ({"extensions": [SIGNED_SENDER_URI]}, "extensions"),
        ({"now_ts": NOW + 301}, "timestamp"),
        ({"now_ts": NOW - 301}, "timestamp"),
    ],
)
@pytest.mark.asyncio
async def test_route_operation_extension_and_clock_binding(
    identity, request_message, change, diagnostic
):
    header = _sign(identity, request_message)
    with pytest.raises(NativeSenderProofError, match=diagnostic):
        await _verify(identity, request_message, [header], _ForbiddenStore(), **change)


@pytest.mark.parametrize(
    "headers",
    [
        [],
        ["one", "two"],
        [""],
        ["x" * 8193],
        ["é"],
        ["a.b"],
        ["a.b.c.d"],
        ["=.b.c"],
        ["a.b.c"],
    ],
)
@pytest.mark.asyncio
async def test_malformed_or_duplicate_header_never_reaches_storage(
    identity, request_message, headers
):
    with pytest.raises(NativeSenderProofError):
        await _verify(identity, request_message, headers, _ForbiddenStore())


@pytest.mark.parametrize(
    "target",
    [
        "classical",
        "pq",
        "kid",
        "sender",
        "domain",
        "version",
        "unknown_claim",
        "noncanonical",
        "duplicate_claim",
    ],
)
@pytest.mark.asyncio
async def test_proof_tampering_and_canonical_structure_are_rejected(
    identity, request_message, target
):
    segments = _sign(identity, request_message).split(".")
    if target in {"classical", "pq"}:
        index = 1 if target == "classical" else 2
        raw = _decode(segments[index])
        segments[index] = _encode(bytes([raw[0] ^ 1]) + raw[1:])
    else:
        claims = json.loads(_decode(segments[0]))
        if target == "kid":
            claims["signers"][0]["kid"] = "other-key"
        elif target == "sender":
            claims["sender"] = "did:web:attacker.example"
        elif target == "domain":
            claims["extension"] = "https://attacker.example/proof"
        elif target == "version":
            claims["version"] = 2
        elif target == "unknown_claim":
            claims["unexpected"] = True
        encoded = rfc8785.dumps(claims)
        if target == "noncanonical":
            encoded += b" "
        elif target == "duplicate_claim":
            encoded = encoded[:-1] + b',"version":1}'
        # A malformed/cross-domain claim can have a genuine signature. Its
        # rejection must not depend on an incidental signature mismatch.
        segments = _resign(identity, encoded).split(".")
    with pytest.raises(NativeSenderProofError):
        await _verify(
            identity, request_message, [".".join(segments)], _ForbiddenStore()
        )


@pytest.mark.parametrize(
    "failure",
    [OSError("storage unavailable"), asyncio.CancelledError("owner canceled")],
)
@pytest.mark.asyncio
async def test_replay_storage_failure_and_cancellation_propagate(
    identity, request_message, failure
):
    header = _sign(identity, request_message)

    class FailedStore:
        async def reserve(self, *args, **kwargs):
            raise failure

    with pytest.raises(type(failure)) as raised:
        await _verify(identity, request_message, [header], FailedStore())
    assert raised.value is failure


@pytest.mark.parametrize(
    "operation,params",
    [
        ("CancelTask", GetTaskRequest(id="t")),
        ("GetTask", SendMessageRequest()),
        ("NotAnOperation", GetTaskRequest(id="t")),
    ],
)
def test_native_operation_and_request_type_cannot_diverge(operation, params):
    with pytest.raises(NativeSenderProofError, match="request type"):
        native_request_digest(params, operation)


def test_signing_rejects_inactive_extension_and_small_header_budget(
    identity, request_message
):
    with pytest.raises(NativeSenderProofError, match="activated"):
        _sign(identity, request_message, extensions=[])
    with pytest.raises(NativeSenderProofError, match="header limit"):
        _sign(
            identity, request_message, policy=NativeProofPolicy(max_header_bytes=1024)
        )


@pytest.mark.asyncio
async def test_concurrent_workers_cannot_both_consume_the_same_nonce(
    identity, request_message, tmp_path
):
    path = str(tmp_path / "concurrent-native-replay.db")
    first = await AsyncDatabase.sqlite(path)
    second = await AsyncDatabase.sqlite(path)
    try:
        stores = [SharedReplayNonceStore(first), SharedReplayNonceStore(second)]
        for store in stores:
            await store.ensure_table()
        header = _sign(identity, request_message)
        outcomes = await asyncio.gather(
            *(_verify(identity, request_message, [header], store) for store in stores),
            return_exceptions=True,
        )
        assert sum(not isinstance(value, BaseException) for value in outcomes) == 1, (
            outcomes
        )
        errors = [value for value in outcomes if isinstance(value, BaseException)]
        assert len(errors) == 1 and isinstance(errors[0], NativeSenderProofError), (
            outcomes
        )
        assert "consumed" in str(errors[0])
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_future_skew_reservation_covers_the_entire_freshness_window(
    identity, request_message, tmp_path
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "native-future-replay.db"))
    try:
        header = _sign(
            identity, request_message, timestamp=NOW + POLICY.max_age_seconds
        )
        await _verify(identity, request_message, [header], SharedReplayNonceStore(db))
        with pytest.raises(NativeSenderProofError, match="consumed"):
            await _verify(
                identity,
                request_message,
                [header],
                SharedReplayNonceStore(db),
                now_ts=NOW + 2 * POLICY.max_age_seconds,
            )
    finally:
        await db.close()


@pytest.mark.parametrize("target", ["algorithm", "timestamp"])
@pytest.mark.asyncio
async def test_unsupported_claims_fail_before_did_resolution(
    identity, request_message, target
):
    segments = _sign(identity, request_message).split(".")
    claims = json.loads(_decode(segments[0]))
    if target == "algorithm":
        claims["signers"][0]["alg"] = "unknown"
        encoded = rfc8785.dumps(claims)
    else:
        # This is valid JSON, but outside RFC 8785's safe integer domain.
        claims["timestamp"] = 2**60
        encoded = json.dumps(claims).encode()
    segments[0] = _encode(encoded)

    def forbidden_resolver(did):
        raise AssertionError("Unsupported proof reached DID resolution")

    with pytest.raises(NativeSenderProofError):
        await _verify(
            identity,
            request_message,
            [".".join(segments)],
            _ForbiddenStore(),
            resolver=forbidden_resolver,
        )


@pytest.mark.asyncio
async def test_required_native_fields_are_checked_before_verification(identity):
    from kestrel_sdk.native_a2a.utils.errors import InvalidParamsError

    with pytest.raises(InvalidParamsError):
        _sign(identity, SendMessageRequest())


@pytest.mark.parametrize("document", [None, [], {"id": "did:web:other.example"}])
@pytest.mark.asyncio
async def test_missing_or_mismatched_did_document_cannot_authenticate(
    identity, request_message, document
):
    header = _sign(identity, request_message)
    with pytest.raises(NativeSenderProofError, match="DID document"):
        await _verify(
            identity,
            request_message,
            [header],
            _ForbiddenStore(),
            resolver=lambda did: document,
        )


@pytest.mark.asyncio
async def test_document_id_must_match_even_when_its_keys_verify(
    identity, request_message
):
    header = _sign(identity, request_message)
    document = {**identity[1], "id": "did:web:other.example"}
    with pytest.raises(NativeSenderProofError, match="mismatched"):
        await _verify(
            identity,
            request_message,
            [header],
            _ForbiddenStore(),
            resolver=lambda did: document,
        )


@pytest.mark.asyncio
async def test_noncanonical_base64_pad_bits_are_rejected_with_unchanged_signature(
    identity, request_message
):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    segments = _sign(identity, request_message).split(".")
    original = segments[1]
    assert len(original) % 4 == 2
    segments[1] = original[:-1] + alphabet[alphabet.index(original[-1]) + 1]
    assert _decode(segments[1]) == _decode(original)
    with pytest.raises(NativeSenderProofError, match="Noncanonical"):
        await _verify(
            identity, request_message, [".".join(segments)], _ForbiddenStore()
        )


@pytest.mark.parametrize("nonce", ["", "x" * 32, "a" * 31])
@pytest.mark.asyncio
async def test_signed_invalid_nonce_cannot_reach_shared_store(
    identity, request_message, nonce
):
    header = _sign(identity, request_message)
    claims = json.loads(_decode(header.split(".")[0]))
    claims["nonce"] = nonce
    malformed = _resign(identity, rfc8785.dumps(claims))
    with pytest.raises(NativeSenderProofError, match="claims"):
        await _verify(identity, request_message, [malformed], _ForbiddenStore())


@pytest.mark.asyncio
async def test_async_resolver_and_resolver_failure_are_preserved(
    identity, request_message, tmp_path
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "async-resolver.db"))
    try:
        header = _sign(identity, request_message)

        async def resolver(did):
            assert did == SENDER
            return identity[1]

        assert (
            await _verify(
                identity,
                request_message,
                [header],
                SharedReplayNonceStore(db),
                resolver=resolver,
            )
        ).sender == SENDER
        failure = OSError("DID resolver unavailable")

        async def failed_resolver(did):
            raise failure

        with pytest.raises(OSError) as raised:
            await _verify(
                identity,
                request_message,
                [header],
                _ForbiddenStore(),
                resolver=failed_resolver,
            )
        assert raised.value is failure
    finally:
        await db.close()


def test_nonnative_parameters_and_unsupported_v1_key_algorithm_fail(identity):
    from kestrel_sovereign.security.crypto_suite import ALG_ECDSA_SECP256K1_SHA256

    with pytest.raises(TypeError, match="official protobuf"):
        native_request_digest(object(), "GetTask")
    pair = generate_hybrid_keypair(classical_alg=ALG_ECDSA_SECP256K1_SHA256)
    with pytest.raises(NativeSenderProofError, match="requires Ed25519"):
        _sign((pair, {}), GetTaskRequest(id="task"), operation="GetTask")


@pytest.mark.asyncio
async def test_duplicate_genuine_headers_are_rejected(identity, request_message):
    header = _sign(identity, request_message)
    with pytest.raises(NativeSenderProofError, match="Exactly one"):
        await _verify(identity, request_message, [header, header], _ForbiddenStore())


@pytest.mark.asyncio
async def test_receiver_checks_its_own_header_limit_before_resolution(
    identity, request_message
):
    header = _sign(identity, request_message)

    def forbidden_resolver(did):
        raise AssertionError("Oversize proof reached DID resolution")

    with pytest.raises(NativeSenderProofError, match="header limit"):
        await _verify(
            identity,
            request_message,
            [header],
            _ForbiddenStore(),
            policy=NativeProofPolicy(max_header_bytes=1024),
            resolver=forbidden_resolver,
        )


@pytest.mark.parametrize("nonce", ["", "x" * 32, "a" * 31])
def test_producer_rejects_invalid_claims_with_typed_proof_error(
    identity, request_message, nonce
):
    with pytest.raises(NativeSenderProofError, match="claims"):
        _sign(identity, request_message, nonce=nonce)
